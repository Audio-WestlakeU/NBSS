import os

os.environ["OMP_NUM_THREADS"] = str(1)  # 使用cpu生成rir时，用到了多进程加速，因此不需要这个
# os.environ["CUDA_VISIBLE_DEVICES"] = str(4)  # choose the gpu to use

import json
from typing import *
from jsonargparse import ArgumentParser
import numpy as np
import tqdm
from numpy.linalg import norm
from numpy.random import uniform
import multiprocessing
from functools import partial
import warnings
import inspect
from scipy.optimize import minimize
from pathlib import Path
# uncomment rir_generator if you use cpu to generate rir
# import rir_generator

# comment gpuRIR if you use cpu to generate rir
import gpuRIR

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)


def estimate_minimal_RT60(room_sz: Union[List[float], np.ndarray]) -> float:
    V = 1.0
    for v in room_sz:
        V = V * v
    S = (room_sz[0] * room_sz[1] + room_sz[0] * room_sz[2] + room_sz[1] * room_sz[2]) * 2
    RT60 = 0.161 * V / S
    return RT60


def is_valid_RT60_for_room(room_sz: Union[List[float], np.ndarray], RT60: float, eps: float = 1e-4) -> bool:
    RT60m = estimate_minimal_RT60(room_sz)
    if RT60 < RT60m + eps:
        return False
    else:
        return True


def is_valid_beta(beta: Union[List[float], np.ndarray]) -> bool:
    return not np.isclose(beta, 0).any()


def beta_SabineEstimation(room_sz, T60, abs_weights=[1.0] * 6):
    '''  Estimation of the reflection coefficients needed to have the desired reverberation time. (The code is taken from gpuRIR)
    
	Parameters
	----------
	room_sz : 3 elements list or numpy array
		Size of the room (in meters).
	T60 : float
		Reverberation time of the room (seconds to reach 60dB attenuation).
	abs_weights : array_like with 6 elements, optional
		Absoprtion coefficient ratios of the walls (the default is [1.0]*6).
	
	Returns
	-------
	ndarray with 6 elements
		Reflection coefficients of the walls as $[beta_{x0}, beta_{x1}, beta_{y0}, beta_{y1}, beta_{z0}, beta_{z1}]$,
		where $beta_{x0}$ is the coeffcient of the wall parallel to the x axis closest
		to the origin of coordinates system and $beta_{x1}$ the farthest.
	'''

    def t60error(x, T60, room_sz, abs_weights):
        alpha = x * abs_weights
        Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
         (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
         (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
        V = np.prod(room_sz)
        if Sa == 0:
            return T60 - 0  # Anechoic chamber
        return abs(T60 - 0.161 * V / Sa)  # Sabine's formula

    abs_weights /= np.array(abs_weights).max()
    result = minimize(t60error, 0.5, args=(T60, room_sz, abs_weights), bounds=[[0, 1]])
    return np.sqrt(1 - result.x * abs_weights).astype(np.float32), result.fun


def generate_rir_cpu(
    room_sz: Union[List[float], np.ndarray],
    pos_src: Union[List[List[float]], np.ndarray],
    pos_rcv: Union[List[List[float]], np.ndarray],
    RT60: float,
    fs: int,
    beta: Optional[np.ndarray] = None,
    sensor_orientations=None,
    sensor_directivity=None,
    sound_velocity: float = 343,
):
    if len(pos_src) == 0:
        return None

    assert RT60 >= 0, RT60
    filter_length: int = int((RT60 + 0.1) * fs)

    room_sz = np.array(room_sz)
    pos_src = np.array(pos_src)
    pos_rcv = np.array(pos_rcv)

    if np.ndim(pos_src) == 1:
        pos_src = np.reshape(pos_src, (1, -1))
    if np.ndim(room_sz) == 1:
        room_sz = np.reshape(room_sz, (1, -1))
    if np.ndim(pos_rcv) == 1:
        pos_rcv = np.reshape(pos_rcv, (1, -1))

    assert room_sz.shape == (1, 3)
    assert pos_src.shape[1] == 3
    assert pos_rcv.shape[1] == 3

    n_src = pos_src.shape[0]
    n_mic = pos_rcv.shape[0]

    if sensor_orientations is None:
        sensor_orientations = np.zeros((2, n_src))
    else:
        raise NotImplementedError(sensor_orientations)

    if sensor_directivity is None:
        sensor_directivity = 'omnidirectional'
    else:
        raise NotImplementedError(sensor_directivity)

    assert filter_length is not None
    rir = np.zeros((n_src, n_mic, filter_length), dtype=np.float64)
    import rir_generator
    for k in range(n_src):
        temp = rir_generator.generate(
            c=sound_velocity,
            fs=fs,
            r=np.ascontiguousarray(pos_rcv),
            s=np.ascontiguousarray(pos_src[k, :]),
            L=np.ascontiguousarray(room_sz[0, :]),
            beta=beta,
            reverberation_time=RT60,
            nsample=filter_length,
            mtype=rir_generator.mtype.omnidirectional,
        )
        rir[k, :, :] = np.asarray(temp.T)

    assert rir.shape[0] == n_src
    assert rir.shape[1] == n_mic
    assert rir.shape[2] == filter_length

    assert not np.any(np.isnan(rir)), f"{np.sum(np.isnan(rir))} values of {rir.size} are NaN."
    return rir


def generate_rir_gpu(
    room_sz: Union[List[float], np.ndarray],
    pos_src: Union[List[List[float]], np.ndarray],
    pos_rcv: Union[List[List[float]], np.ndarray],
    RT60: float,
    fs: int,
    sound_velocity: float = 343,
    att_diff: float = None,
    beta: Optional[np.ndarray] = None,
):
    if len(pos_src) == 0:
        return None

    if RT60 == 0:  # direct-path rir
        RT60 = 1  # 随便给一个值，防止出错
        Tmax = 0.1
        nb_img = [1, 1, 1]
        beta = [0] * 6
        Tdiff = None
    else:
        Tmax = gpuRIR.att2t_SabineEstimator(60.0, RT60)
        if att_diff is not None:
            Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, RT60)
            nb_img = gpuRIR.t2n(Tdiff, room_sz)
        else:
            Tdiff = None
            nb_img = gpuRIR.t2n(Tmax, room_sz)

        if beta is None:
            beta = gpuRIR.beta_SabineEstimation(room_sz, RT60)  # reflection coefficients

            if is_valid_beta(beta) == False:
                warnings.warn(f'beta is invalid for gpuRIR, which might indicate the given RT60={RT60} could not achieved with the given room_sz={room_sz}')

    rir = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=pos_src, pos_rcv=pos_rcv, nb_img=nb_img, Tmax=Tmax, Tdiff=Tdiff, fs=fs, c=sound_velocity)

    return rir


def normalize(vec: np.ndarray) -> np.ndarray:
    # get unit vector
    vec = vec / norm(vec)
    vec = vec / norm(vec)
    assert np.isclose(norm(vec), 1), 'norm of vec is not close to 1'
    return vec


def plot_room(room_sz: Union[List[float], np.ndarray], pos_src: np.ndarray, pos_rcv: np.ndarray, pos_noise: np.ndarray, saveto: str = None) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
    plt.close('all')
    fig = plt.figure(figsize=(10, 8))
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.scatter(pos_rcv[:, 0], pos_rcv[:, 1], pos_rcv[:, 2])
    if len(pos_rcv) > 2:
        # draw the first half mics with different color for checking the rotation
        ax.scatter(pos_rcv[:len(pos_rcv) // 2, 0], pos_rcv[:len(pos_rcv) // 2, 1], pos_rcv[:len(pos_rcv) // 2, 2], c='r')
    ax.scatter(pos_src[:, 0], pos_src[:, 1], pos_src[:, 2])
    if pos_noise is not None and len(pos_noise) > 0:
        ax.scatter(pos_noise[:, 0], pos_noise[:, 1], pos_noise[:, 2])

    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_xlim3d([0, room_sz[0]])
    ax.set_ylim3d([0, room_sz[1]])
    ax.set_zlim3d([0, room_sz[2]])
    plt.show(block=True)
    if saveto is not None:
        plt.savefig("config/rirs/images/" + saveto + '.jpg')
    plt.close()


def circular_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    # 生成圆阵的拓扑（原点为中心），后期可以通过旋转、改变中心的位置来实现阵列位置的改变
    pos_rcv = np.empty((mic_num, 3))
    v1 = np.array([1, 0, 0])  # 第一个麦克风的位置（要求单位向量）
    v1 = normalize(v1)  # 单位向量
    # 将v1绕原点水平旋转angle角度，来生成其他mic的位置
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
    for idx, angle in enumerate(angles):
        x = v1[0] * np.cos(angle) - v1[1] * np.sin(angle)
        y = v1[0] * np.sin(angle) + v1[1] * np.cos(angle)
        pos_rcv[idx, :] = normalize(np.array([x, y, 0]))
    # 设置radius
    pos_rcv *= radius
    return pos_rcv


def linear_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    xs = np.arange(start=0, stop=radius * mic_num, step=radius)
    xs -= np.mean(xs)  # 将中心移动到原点
    pos_rcv = np.zeros((mic_num, 3))
    pos_rcv[:, 0] = xs
    return pos_rcv


def chime3_array_geometry() -> np.ndarray:
    # TODO 加入麦克风的朝向向量，以及麦克风的全向/半向
    pos_rcv = np.zeros((6, 3))
    pos_rcv[0, :] = np.array([-0.1, 0.095, 0])
    pos_rcv[1, :] = np.array([0, 0.095, 0])
    pos_rcv[2, :] = np.array([0.1, 0.095, 0])
    pos_rcv[3, :] = np.array([-0.1, -0.095, 0])
    pos_rcv[4, :] = np.array([0, -0.095, 0])
    pos_rcv[5, :] = np.array([0.1, -0.095, 0])

    # 验证边长是否正确，边与边之间是否垂直
    assert np.isclose(np.linalg.norm(pos_rcv[0, :] - pos_rcv[1, :]), 0.1), 'distance between #1 and #2 is wrong'
    assert np.isclose(np.linalg.norm(pos_rcv[1, :] - pos_rcv[2, :]), 0.1), 'distance between #2 and #3 is wrong'
    assert np.isclose(np.linalg.norm(pos_rcv[0, :] - pos_rcv[3, :]), 0.19), 'distance between #1 and #4 is wrong'
    assert np.isclose(np.linalg.norm(pos_rcv[2, :] - pos_rcv[5, :]), 0.19), 'distance between #3 and #6 is wrong'
    assert np.isclose(np.linalg.norm(pos_rcv[3, :] - pos_rcv[4, :]), 0.1), 'distance between #4 and #5 is wrong'
    assert np.isclose(np.linalg.norm(pos_rcv[4, :] - pos_rcv[5, :]), 0.1), 'distance between #5 and #6 is wrong'
    assert np.isclose(np.dot(pos_rcv[0, :] - pos_rcv[1, :], pos_rcv[0, :] - pos_rcv[3, :]), 0), 'not vertical'
    assert np.isclose(np.dot(pos_rcv[2, :] - pos_rcv[5, :], pos_rcv[4, :] - pos_rcv[5, :]), 0), 'not vertical'
    return pos_rcv


def libricss_array_geometry() -> np.ndarray:
    pos_rcv = np.zeros((7, 3))
    pos_rcv_c = circular_array_geometry(radius=0.0425, mic_num=6)
    pos_rcv[1:, :] = pos_rcv_c
    return pos_rcv


def rotate(pos_rcv: np.ndarray, x_angle: Optional[float] = None, y_angle: Optional[float] = None, z_angle: Optional[float] = None) -> np.ndarray:
    # 将以原点为中心的麦克风分别绕X、Y、Z轴旋转给定角度(单位：rad)
    def _rotate(pos_rcv: np.ndarray, angle: float, dims: Tuple[int, int]) -> np.ndarray:
        assert len(set(dims)) == 2, "dims参数应该给两个不同的值"
        pos_rcv_new = np.empty_like(pos_rcv)
        pos_rcv_new[:, dims[0]] = pos_rcv[:, dims[0]] * np.cos(angle) - pos_rcv[:, dims[1]] * np.sin(angle)
        pos_rcv_new[:, dims[1]] = pos_rcv[:, dims[0]] * np.sin(angle) + pos_rcv[:, dims[1]] * np.cos(angle)
        dim2 = list({0, 1, 2} - set(dims))[0]
        pos_rcv_new[:, dim2] = pos_rcv[:, dim2]  # 旋转轴的值在旋转前后应该是相等的

        # check angle
        norm2d = norm(pos_rcv[:, dims], axis=-1)
        real_angles = []
        for i, n in enumerate(norm2d):
            if n == 0:
                real_angles.append(None)
            else:
                real_angles.append(np.arccos(np.clip((pos_rcv[i, dims] * pos_rcv_new[i, dims]).sum(axis=-1) / (n**2), a_min=-1, a_max=1)))
        # move angle to range [-pi, pi]
        angle = angle % (2 * np.pi)
        for i, ra in enumerate(real_angles):
            if np.isclose(norm2d[i], 0):
                continue  # skip angle check if the point is close to orgin
            assert np.isclose(ra, angle) or np.isclose(ra + angle, np.pi * 2), (ra, angle)
        # check relative distance
        dist_old = norm(pos_rcv[:, np.newaxis, :] - pos_rcv[np.newaxis, :, :], axis=-1)  # shape [M, M]
        dist_new = norm(pos_rcv_new[:, np.newaxis, :] - pos_rcv_new[np.newaxis, :, :], axis=-1)  # shape [M, M]
        assert np.allclose(dist_old, dist_new)
        return pos_rcv_new

    for ang, dims in zip([x_angle, y_angle, z_angle], [(1, 2), (2, 0), (0, 1)]):
        if ang is not None:
            pos_rcv = _rotate(pos_rcv=pos_rcv, angle=ang, dims=dims)
    return pos_rcv


def generate_rir_cfg_list(
        index: Optional[int] = None,
        spk_num: int = 1,
        noise_num: int = 0,
        room_size_lims: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((3, 8), (3, 8), (3, 4)),
        mic_zlim: Tuple[float, float] = (1.0, 1.5),
        spk_zlim: Tuple[float, float] = (1.0, 1.8),
        RT60_lim: Tuple[float, float] = (0.1, 0.6),
        rir_nums: Tuple[int, int, int] = (40000, 5000, 3000),
        arr_geometry: Union[Literal['circular'], Literal['linear'], Literal['chime3'], Literal['libricss']] = 'libricss',
        arr_radius: Tuple[float, float] = (0.1, 0.1),
        arr_rotate_lims: Union[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]], Literal['auto']] = 'auto',
        arr_room_center_dist: Union[float, Literal['auto']] = 'auto',
        wall_abs_weights_lims: Union[List[Tuple[float, float]], Literal['auto'], Literal[None]] = 'auto',
        mic_num: int = 6,
        mic_pos_var: float = 0,
        spk_arr_dist: Union[Tuple[float, float], Literal['auto'], Literal['random']] = 'auto',
        fs: int = 16000,
        attn_diff: Tuple[Optional[float], Optional[float]] = (None, None),
        save_to: Union[Literal['auto'], str] = 'auto',
        rir_dir: str = 'dataset/rirs_generated',
        seed: int = 2023,
):
    """configuration file generation

    Args:
        index: the index of one sample, please give None for this parameter
        spk_num: the number of speakers.
        noise_num: the number of point noises.
        room_size_lims: the x, y, z range of room.
        mic_zlim: the z range of microphone center.
        spk_zlim: the height range of speaker.
        RT60_lim: the range of RT60.
        rir_nums: the number of training/validation/test set rirs.
        arr_geometry: 'circular', 'linear' or 'chime3'.
        arr_radius: the range of radius of array.
        arr_rotate_lims: rotation angle range for x/y/z axis.
        arr_room_center_dist: the max distance between the center of array and room
        wall_abs_weights_lims: the weights of wall absorbtion coefficients. TODO: add half-open, 
        mic_num: the number of microphones.
        mic_pos_var: microphone array position variation (m).
        spk_arr_dist: the distance range between the center of microphone array and speaker.
        fs: sample rate.
        attn_diff: starts from what attenuation (dB) to use diffuse model to generate rir for speech and noise. diffuse model will speed up the simulation but is not accurate?
        save_to: save the configuration file to.
        rir_dir: the dir to save generated rirs
        seed: the random seeds.
    """
    if index is None:
        # set parameters and start multiprocessing generation
        assert arr_geometry in ['circular', 'linear', 'chime3', 'libricss'], "only supports circular, linear, chime3 and libricss array for now"

        if arr_geometry == 'circular' or arr_geometry == 'linear':
            if arr_rotate_lims == 'auto':
                arr_rotate_lims = (None, None, (0, 2 * np.pi))  # rotate by z-axis only
            if spk_arr_dist == 'auto':
                spk_arr_dist = 'random'
            if arr_room_center_dist == 'auto':
                arr_room_center_dist = 0.5
        elif arr_geometry == 'chime3':
            if arr_rotate_lims == 'auto':
                arr_rotate_lims = ((0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi))
            if spk_arr_dist == 'auto':
                spk_arr_dist = (0.3, 0.5)
            if arr_room_center_dist == 'auto':
                arr_room_center_dist = 2.0
        elif arr_geometry == 'libricss':
            arr_radius = (0.0425, 0.0425)
            mic_num = 7
            if arr_rotate_lims == 'auto':
                arr_rotate_lims = (None, None, (0, 2 * np.pi))  # rotate by z-axis only
            if spk_arr_dist == 'auto':
                spk_arr_dist = (0.5, 4.5)
            if arr_room_center_dist == 'auto':
                arr_room_center_dist = 1.0

        if wall_abs_weights_lims == 'auto':
            wall_abs_weights_lims = [(0.5, 1.0)] * 6
        elif wall_abs_weights_lims is None:
            wall_abs_weights_lims = [(1.0, 1.0)] * 6
        else:
            assert len(wall_abs_weights_lims) == 6, "you should give the weights of six walls"
        if save_to == 'auto':
            save_to = os.path.join(rir_dir, 'rir_cfg.npz')
        args = locals().copy()  # capture the parameters passed to this function or their edited values

        if os.path.exists(save_to):
            cfg = dict(np.load(save_to, allow_pickle=True))
            print('load rir cfgs from file ' + save_to)
            print('Args in npz: \n', cfg['args'].item())
            return cfg
        else:
            print('Args:')
            print(dict(args), '\n')

        rir_num = sum(rir_nums)
        print('generating rir cfgs. ', end=' ')
        import time
        ts = time.time()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2) as p:
            new_args = args.copy()
            del new_args['index']
            rir_pars = p.map(partial(generate_rir_cfg_list, **new_args), range(rir_num))
        print('time used: ', time.time() - ts)

        cfg = {
            'args': np.array(args),
            'rir_pars': rir_pars,
        }

        # save to npz
        dir = os.path.dirname(save_to)
        if len(dir) > 0:
            os.makedirs(dir, exist_ok=True)
        np.savez_compressed(save_to, **cfg)
        return cfg

    # generate one room cfg
    np.random.seed(seed=seed + index)
    xlim, ylim, zlim = room_size_lims

    # sample radius / RT60 / room_sz / abs_weights
    array_r_this = uniform(*arr_radius)
    RT60 = uniform(*RT60_lim)  # sample a RT60
    room_sz = [uniform(*xlim), uniform(*ylim), uniform(*zlim)]  # sample a room
    # resample if the RT60 could not be satisfied in this room
    while is_valid_RT60_for_room(room_sz, RT60) == False:  # or is_valid_beta(beta):
        room_sz = [uniform(*xlim), uniform(*ylim), uniform(*zlim)]
        RT60 = uniform(*RT60_lim)
    # sample abs_weights then compute reflection coefficients
    abs_weights = [uniform(*abs_lim) for abs_lim in wall_abs_weights_lims]
    beta, t60error = beta_SabineEstimation(room_sz, RT60, abs_weights=abs_weights)
    while t60error > 0.05:
        abs_weights = [uniform(*abs_lim) for abs_lim in wall_abs_weights_lims]
        beta, t60error = beta_SabineEstimation(room_sz, RT60, abs_weights=abs_weights)
    # if not is_valid_beta(beta):  #  t60error > 0.05:
    #     warnings.warn(f'the given RT60={RT60} could not achieved with the given room_sz={room_sz} and abs_weights={abs_weights}')

    # microphone positions
    mic_center = None
    while mic_center is None or mic_center[0] < 0.5 or mic_center[1] < 0.5 or mic_center[0] > room_sz[0] - 0.5 or mic_center[1] > room_sz[1] - 0.5:
        mic_center = np.array([
            uniform(room_sz[0] / 2 - arr_room_center_dist, room_sz[0] / 2 + arr_room_center_dist),
            uniform(room_sz[1] / 2 - arr_room_center_dist, room_sz[1] / 2 + arr_room_center_dist),
            uniform(*mic_zlim),
        ])
    if arr_geometry == 'circular':
        pos_rcv = circular_array_geometry(radius=array_r_this, mic_num=mic_num)
    elif arr_geometry == 'linear':
        pos_rcv = linear_array_geometry(radius=array_r_this, mic_num=mic_num)
    elif arr_geometry == 'chime3':
        pos_rcv = chime3_array_geometry()
    else:
        assert arr_geometry == 'libricss', arr_geometry
        pos_rcv = libricss_array_geometry()

    # rotate the array by x/y/z axis
    x_angle, y_angle, z_angle = [None if lim is None else uniform(*lim) for lim in arr_rotate_lims]
    pos_rcv = rotate(pos_rcv=pos_rcv, x_angle=x_angle, y_angle=y_angle, z_angle=z_angle)

    # move center from origin to mic_center
    pos_rcv += mic_center[np.newaxis, :]

    # add small position variations to the (x,y,z) of each mic for simulating position's inperfection
    if mic_pos_var > 0:
        pos_rcv = pos_rcv + uniform(low=-mic_pos_var, high=mic_pos_var, size=pos_rcv.shape)

    # sample speaker postions
    pos_src = np.empty((spk_num, 3))
    # all speaker's loc are randomly sampled
    for iiii in range(0, spk_num):
        pos_src[iiii, :] = uniform(0.5, room_sz[0] - 0.5), uniform(0.5, room_sz[1] - 0.5), uniform(spk_zlim[0], spk_zlim[1])
        if spk_arr_dist == 'random':
            continue
        while norm(pos_src[iiii, :] - mic_center) < spk_arr_dist[0] or norm(pos_src[iiii, :] - mic_center) > spk_arr_dist[1]:
            # if the spk_mic_dis requirements are not satisfied, then resample a position
            pos_src[iiii, :] = uniform(0.5, room_sz[0] - 0.5), uniform(0.5, room_sz[1] - 0.5), uniform(spk_zlim[0], spk_zlim[1])

    # generate the positions of noise sources
    pos_noise = []
    for iiii in range(noise_num):
        pos_noise.append([uniform(0.1, sz - 0.1) for sz in room_sz])
    pos_noise = np.array(pos_noise)

    # plot room
    # print(x_angle / np.pi * 180, y_angle / np.pi * 180, z_angle / np.pi * 180)
    # plot_room(room_sz=room_sz, pos_src=pos_src, pos_rcv=pos_rcv, pos_noise=pos_noise, saveto=None)

    par = {
        'index': index,
        'RT60': RT60,
        'room_sz': room_sz,
        'pos_src': pos_src.astype(np.float32),
        'pos_rcv': pos_rcv.astype(np.float32),
        'pos_noise': pos_noise.astype(np.float32),
        # 'abs_weights': abs_weights,
        'beta': beta,
    }
    return par


def generate_rir_files(rir_cfg: Dict[str, Any], rir_dir: str, rir_nums: Tuple[int, int, int], use_gpu: bool):
    train_rir_num, val_rir_num, test_rir_num = rir_nums

    pars = rir_cfg['rir_pars']
    fs = rir_cfg['args'].item()['fs']
    attn_diff_speech = rir_cfg['args'].item()['attn_diff']
    attn_diff_noise = None
    if isinstance(attn_diff_speech, tuple):
        attn_diff_noise = attn_diff_speech[1]
        attn_diff_speech = attn_diff_speech[0]

    if (Path(rir_dir) / 'train').exists() or (Path(rir_dir) / 'validation').exists() or (Path(rir_dir) / 'test').exists():
        ans = input("dir " + rir_dir + ' exists, still generate rir?(y/n)')
        assert ans in ['y', 'n'], ans
        if ans == 'n':
            return
    else:
        os.makedirs(rir_dir, exist_ok=True)

    for setdir in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(rir_dir, setdir), exist_ok=True)

    def __gen__(par, fs, use_gpu: bool):
        index = par['index']
        RT60 = par['RT60']
        room_sz = par['room_sz']
        pos_src = np.array(par['pos_src'])
        pos_rcv = np.array(par['pos_rcv'])
        pos_noise = np.array(par['pos_noise'])
        # abs_weights = np.array(par['abs_weights']) if 'abs_weights' in par else None
        beta = np.array(par['beta']) if 'beta' in par else None

        if index < train_rir_num:
            setdir = 'train'
        elif index >= train_rir_num and index < train_rir_num + val_rir_num:
            setdir = 'validation'
        else:
            setdir = 'test'
        save_to = os.path.join(rir_dir, setdir, str(index) + '.npz')
        if os.path.exists(save_to):
            try:  # try to load, if no error is reported then skip this
                np.load(save_to, allow_pickle=True)
                return
            except:
                ...
        if not use_gpu:
            rir = generate_rir_cpu(room_sz, pos_src, pos_rcv, RT60, fs, beta=beta)  # reverbrant rir
            rir_dp = generate_rir_cpu(room_sz, pos_src, pos_rcv, 0, fs)  # direct path rir
            rir_noise = generate_rir_cpu(room_sz, pos_noise, pos_rcv, RT60, fs, beta=beta)  # noise rir
        else:
            rir = generate_rir_gpu(room_sz, pos_src, pos_rcv, RT60, fs, att_diff=attn_diff_speech, beta=beta)  # reverbrant rir
            rir_dp = generate_rir_gpu(room_sz, pos_src, pos_rcv, 0, fs)  # direct path rir
            rir_noise = generate_rir_gpu(room_sz, pos_noise, pos_rcv, RT60, fs, att_diff=attn_diff_noise, beta=beta)  # noise rir, uses diffuse model after 20 dB attenuation
        if rir_noise is not None:
            rir_noise = rir_noise.astype(np.float16)
        np.savez_compressed(save_to, fs=fs, RT60=RT60, room_sz=room_sz, pos_src=pos_src, pos_rcv=pos_rcv, pos_noise=pos_noise, rir=rir, rir_dp=rir_dp, rir_noise=rir_noise)

    if not use_gpu:
        from p_tqdm import p_map
        p_map(
            partial(__gen__, fs=fs, use_gpu=use_gpu),
            pars,
            num_cpus=multiprocessing.cpu_count() // 2,
        )
    else:
        pbar = tqdm.tqdm(total=len(pars))
        pbar.set_description('generating rirs')
        for i, par in enumerate(pars):
            pbar.update()
            __gen__(par=par, fs=fs, use_gpu=use_gpu)


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python generate_rirs.py --help
    parser = ArgumentParser(description='code for generating RIRs by Changsheng Quan @ Westlake University')
    parser.add_function_arguments(generate_rir_cfg_list)  # add_argument for the function generate_rir_cfg_list
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')

    args = parser.parse_args()

    # get paramters for function `generate_rir_cfg_list`
    sig = inspect.signature(generate_rir_cfg_list)
    args_for_generate_rir_cfg_list = dict()
    for param in sig.parameters.values():
        args_for_generate_rir_cfg_list[param.name] = getattr(args, param.name)

    # generate configuration
    rir_cfg = generate_rir_cfg_list(**args_for_generate_rir_cfg_list)

    # generate rir files
    generate_rir_files(
        rir_cfg=rir_cfg,
        rir_dir=args.rir_dir,
        rir_nums=args.rir_nums,
        use_gpu=args.use_gpu,
    )
