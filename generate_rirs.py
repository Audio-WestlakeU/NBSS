import os

os.environ["OMP_NUM_THREADS"] = str(1)  # 使用cpu生成rir时，用到了多进程加速，因此不需要这个
# os.environ["CUDA_VISIBLE_DEVICES"] = str(7)  # choose the gpu to use

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
import multiprocessing as mp


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
    att_max: float = 60.0,
    beta: Optional[np.ndarray] = None,
):
    import gpuRIR
    # importlib.reload(gpuRIR)  # reload gpuRIR to use another gpu
    gpuRIR.activateMixedPrecision(False)
    gpuRIR.activateLUT(False)

    if len(pos_src) == 0:
        return None

    if RT60 == 0:  # direct-path rir
        RT60 = 1  # 随便给一个值，防止出错
        Tmax = 0.1
        nb_img = [1, 1, 1]
        beta = [0] * 6
        Tdiff = None
    else:
        Tmax = gpuRIR.att2t_SabineEstimator(att_max, RT60)
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
    if not isinstance(pos_src, list):
        pos_src = [pos_src]
    for i in range(len(pos_src)):
        ax.scatter(pos_src[i][:, 0], pos_src[i][:, 1], pos_src[i][:, 2])
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


def plot_room_2d(room_sz: Union[List[float], np.ndarray], pos_src: np.ndarray, pos_rcv: np.ndarray, pos_noise: np.ndarray, saveto: str = None) -> None:
    import matplotlib.pyplot as plt
    plt.close('all')
    fig = plt.figure(figsize=(room_sz[0] * 2, room_sz[1] * 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(pos_rcv[:, 0], pos_rcv[:, 1])
    if len(pos_rcv) > 2:
        # draw the first half mics with different color for checking the rotation
        ax.scatter(pos_rcv[:len(pos_rcv) // 2, 0], pos_rcv[:len(pos_rcv) // 2, 1], c='r', s=1)
    if not isinstance(pos_src, list):
        pos_src = [pos_src]
    for i in range(len(pos_src)):
        ax.scatter(pos_src[i][:, 0], pos_src[i][:, 1], s=1)
    if pos_noise is not None and len(pos_noise) > 0:
        ax.scatter(pos_noise[:, 0], pos_noise[:, 1], s=30, c='green')

    ax.set(xlabel="X", ylabel="Y")
    ax.set_xlim([0, room_sz[0]])
    ax.set_ylim([0, room_sz[1]])
    plt.show(block=True)
    # plt.show()
    if saveto is not None:
        plt.savefig("configs/rirs/images/" + saveto + '.jpg')
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


def circular_cm_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    # 圆形阵列+中心麦克风
    # circular array with central microphone
    pos_rcv = np.zeros((mic_num, 3))
    pos_rcv_c = circular_array_geometry(radius=radius, mic_num=mic_num - 1)
    pos_rcv[1:, :] = pos_rcv_c
    return pos_rcv


def audiowu_high_array_geometry() -> np.ndarray:
    # the high-resolution mic array of the audio lab of westlake university
    R = 0.03
    pos_rcv = np.zeros((32, 3))
    pos_rcv[:8, :] = circular_array_geometry(radius=R, mic_num=8)
    pos_rcv[8:16, :] = circular_array_geometry(radius=R * 2, mic_num=8)
    pos_rcv[16:24, :] = circular_array_geometry(radius=R * 3, mic_num=8)
    pos_rcv[25, :] = np.array([R * 4, 0, 0])
    pos_rcv[26, :] = np.array([R * 5, 0, 0])
    pos_rcv[27, :] = np.array([-R * 4, 0, 0])

    L = 0.045
    pos_rcv[28, :] = np.array([0, 0, L])
    pos_rcv[29, :] = np.array([0, 0, L * 2])
    pos_rcv[30, :] = np.array([0, 0, -L])
    pos_rcv[31, :] = np.array([0, 0, -L * 2])
    return pos_rcv


def audiowu_low_array_geometry() -> np.ndarray:
    # the high-resolution mic array of the audio lab of westlake university
    R = 0.03
    L = 0.03
    pos_rcv = np.zeros((16, 3))
    pos_rcv[:8, :] = circular_array_geometry(radius=R, mic_num=8)
    pos_rcv[9, :] = np.array([L * 2, 0, 0])
    pos_rcv[10, :] = np.array([L * 3, 0, 0])
    pos_rcv[11, :] = np.array([L * 4, 0, 0])
    pos_rcv[12, :] = np.array([-L * 2, 0, 0])
    pos_rcv[13, :] = np.array([-L * 3, 0, 0])

    pos_rcv[14, :] = np.array([0, L * 2, 0])
    pos_rcv[15, :] = np.array([0, -L * 2, 0])
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


def generate_4points_sin_trajectory(
        room_sz: Union[List[float], np.ndarray],  # the size of room
        rcv_pos: np.ndarray,  # [N, 3]
        min_src_array_dist: np.ndarray,  # [3]
        min_src_boundary_dist: np.ndarray,  # [3]
        src_z: float,  # the height of source
        desired_dist_pts: float = 0.1,  # the distance between neighbouring points  点之间的间隔
        equal_dist: bool = False,  # neighbouring points have equal distance
        max_ratio: float = 3,  # maximum allowed distantce for not equal case
):
    # 移动声源：
    # 1. 从房间的四个区域分别采用一个点
    # 2. 按照区域将四个点连接起来. 每条连线沿着线方向每10cm采样一个点, 这样仿真超长语音的时候, 刚好可以沿着四个点转圈

    xr, yr, zr = room_sz
    xa, ya, za = min_src_array_dist
    xb, yb, zb = min_src_boundary_dist
    array_center = rcv_pos.mean(axis=0)
    assert za == 0 and zb == 0, "not implemented"
    # step 1: sample four points in four area of the room
    # left-down area
    src_pos_max = array_center + np.array([-xa, -ya, 0])
    src_pos_min = np.array([0, 0, 0]) + np.array([xb, yb, 0])
    src_pos_ld = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
    src_pos_ld[2] = src_z
    # right-down area
    src_pos_max = array_center + np.array([xa, -ya, 0])
    src_pos_min = np.array([xr, 0, 0]) + np.array([-xb, yb, 0])
    src_pos_rd = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
    src_pos_rd[2] = src_z
    # right-top area
    src_pos_max = array_center + np.array([xa, ya, 0])
    src_pos_min = np.array([xr, yr, 0]) + np.array([-xb, -yb, 0])
    src_pos_rt = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
    src_pos_rt[2] = src_z
    # left-top area
    src_pos_max = array_center + np.array([-xa, ya, 0])
    src_pos_min = np.array([0, yr, 0]) + np.array([xb, -yb, 0])
    src_pos_lt = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
    src_pos_lt[2] = src_z

    # step 2: connects the four points in order. For each pair of points, sinusoidal oscilations are added to each point for each axis
    trajs = []
    for src_pos_ini, src_pos_end in [(src_pos_ld, src_pos_rd), (src_pos_rd, src_pos_rt), (src_pos_rt, src_pos_lt), (src_pos_lt, src_pos_ld)]:
        dist_ini_end = np.sqrt(np.sum((src_pos_ini - src_pos_end)**2))
        if equal_dist == False:
            # solve p_end= p_ini + A*sin(w*nb_points+n*pi) + vec_mov*nb_points, so that the last point is p_end
            nb_points = int(dist_ini_end / desired_dist_pts)
            A = np.random.random(3) * np.array([xb, yb, 0])  # Magnitude oscilations with [xb,yb,0]
            w = 2 * np.pi / nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
            vec_mov = ((src_pos_end - src_pos_ini) - A * np.sin(w * nb_points)) / nb_points
            traj_pts = src_pos_ini + vec_mov * np.arange(nb_points)[:, np.newaxis] + A * np.sin(w * np.arange(0, nb_points)[:, np.newaxis])
            while len(traj_pts) > 1 and np.max(norm(traj_pts[1:] - traj_pts[:-1], axis=-1)) > max_ratio * desired_dist_pts:
                A = np.random.random(3) * np.array([xb, yb, 0])  # Magnitude oscilations with [xb,yb,0]
                w = 2 * np.pi / nb_points * np.random.random(3) * 2  # Between 0 and 2 oscilations in each axis
                vec_mov = ((src_pos_end - src_pos_ini) - A * np.sin(w * nb_points)) / nb_points
                traj_pts = src_pos_ini + vec_mov * np.arange(nb_points)[:, np.newaxis] + A * np.sin(w * np.arange(0, nb_points)[:, np.newaxis])
            trajs.append(traj_pts)
        else:
            traj_pts = []
            unit_vec = (src_pos_end - src_pos_ini) / dist_ini_end  # 起点指向终点的单位方向向量
            A = np.random.random(3) * np.array([xb, yb, 0])  # Magnitude oscilations with [xb,yb,0]
            w = 2 * np.pi * np.random.randint(1, 4, size=3)  # Between 1 and 2 times 2pi rad oscilations in each axis
            # 沿unit_vec方向移动多长距离, 才使得移动后点与起点的距离接近于desired_dist_pts
            x_vec = 0.0
            while x_vec < dist_ini_end:
                osc = A * np.sin(w * (x_vec / dist_ini_end))
                p0 = src_pos_ini + unit_vec * x_vec + osc
                traj_pts.append(p0)

                def err_func(x_mov: float, x_vec: float, p0: np.ndarray, desired_dist_pts: float, A: np.ndarray, w: np.ndarray):
                    # minimize the distance between current point and next point
                    osc = A * np.sin(w * ((x_vec + x_mov) / dist_ini_end))
                    p1 = src_pos_ini + unit_vec * (x_vec + x_mov) + osc
                    dd = np.sqrt(np.sum((p1 - p0)**2))
                    return np.abs(dd - desired_dist_pts)

                for factor in [1.0, 1.5, 3]:
                    res = minimize(err_func, x0=[desired_dist_pts / 10], bounds=[(0, desired_dist_pts * factor)], tol=desired_dist_pts / 100, args=(x_vec, p0, desired_dist_pts, A, w))
                    if res.fun < desired_dist_pts / 100:
                        break
                x_vec = x_vec + res.x[0]
                # print(res.x[0], res.fun)
            traj_pts = np.array(traj_pts, dtype=np.float16)  # [N,3]
            trajs.append(traj_pts)
    traj_pts = np.concatenate(trajs, axis=0)
    if (traj_pts >= 0).all() and (traj_pts <= np.array([room_sz])).all():
        return traj_pts, np.stack([src_pos_ld, src_pos_rd, src_pos_rt, src_pos_lt], dtype=np.float16)
    else:
        # assert (traj_pts >= 0).all() and (traj_pts <= np.array([room_sz])).all(), "traj_pts out of the room"
        return None, None  # return None if any points are out of the room


def generate_rir_cfg_list(
    index: Optional[int] = None,
    spk_num: int = 1,
    noise_num: int = 0,
    room_size_lims: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((3, 8), (3, 8), (3, 4)),
    mic_zlim: Tuple[float, float] = (1.0, 1.5),
    spk_zlim: Tuple[float, float] = (1.0, 1.8),
    RT60_lim: Tuple[float, float] = (0.1, 0.6),
    rir_nums: Tuple[int, int, int] = (40000, 5000, 3000),
    arr_geometry: Union[Literal['circular', 'circular+cm', 'linear', 'chime3', 'libricss'], str, List[str]] = 'circular+cm',
    arr_radius: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = (0.1, 0.1),
    arr_rotate_lims: Union[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]], Literal['auto']] = 'auto',
    arr_room_dist: Union[Tuple[float, float], Literal['auto']] = (0.5, 0.2),
    wall_abs_weights_lims: Union[List[Tuple[float, float]], Literal['auto', None]] = 'auto',
    mic_num: Union[int, List[int], Tuple[Union[int, List[int]], int]] = 6,
    mic_pos_var: float = 0,
    spk_arr_dist: Union[Tuple[float, float], Literal['auto', 'random']] = 'auto',
    trajectory: Optional[Tuple[str, float]] = None,
    fs: int = 8000,
    attn_diff: Tuple[Optional[float], Optional[float], Optional[float]] = (15.0, 15.0, 60.0),
    save_to: Union[Literal['auto'], str] = 'auto',
    rir_dir: str = 'dataset/rirs_generated',
    seed: Optional[int] = None,
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
        arr_room_dist: (a, b), where a is the max distance between room center and array center, and b is the min distance between room boundary and array center
        wall_abs_weights_lims: the weights of wall absorbtion coefficients. TODO: add half-open, 
        mic_num: the number of microphones if given int, otherwise [the number of microphones, the number of randomly selected microphones].
        mic_pos_var: microphone array position variation (m).
        spk_arr_dist: the distance range between the center of microphone array and speaker. For trajectory, this parameter is used as the minimum distance between microphone array and speaker.
        trajectory: the trajectory type of each source and the distance between adjacent trajectory points (m). can be `None` or something like (`4points+sin`, 0.05).
        fs: sample rate.
        attn_diff: starts from what attenuation (dB) to use diffuse model to generate rir for speech and noise, and the maximum attenuation (dB). diffuse model will speed up the simulation but is not accurate?
        save_to: save the configuration file to.
        rir_dir: the dir to save generated rirs
        seed: the random seeds.
    """
    if index is None:
        if save_to == 'auto':
            save_to = os.path.join(rir_dir, 'rir_cfg.npz')
        save_to = os.path.expanduser(save_to)

        if os.path.exists(save_to):
            cfg = dict(np.load(save_to, allow_pickle=True))
            print('load rir cfgs from file ' + save_to)
            print('Args in npz: \n', cfg['args'].item())
            return cfg

        # set parameters and start multiprocessing generation
        if isinstance(arr_geometry, str):
            assert arr_geometry in ['circular', 'circular+cm', 'linear', 'chime3', 'libricss', 'audiowu_low', 'audiowu_high'], f"not supported array geometry {arr_geometry}"
        else:
            for arrgmt in arr_geometry:
                assert arrgmt in ['circular', 'circular+cm', 'linear', 'chime3', 'libricss', 'audiowu_low', 'audiowu_high'], f"not supported array geometry {arrgmt}"
            del arrgmt

        assert trajectory is None or trajectory[0] in ['4points+sin', '4points+sin+eqdist'], trajectory

        if arr_geometry == 'circular' or arr_geometry == 'circular+cm' or arr_geometry == 'linear':
            if arr_rotate_lims == 'auto':
                arr_rotate_lims = (None, None, (0, 2 * np.pi))  # rotate by z-axis only
            if spk_arr_dist == 'auto':
                spk_arr_dist = 'random'
            if arr_room_dist == 'auto':
                arr_room_dist = (0.5, 0.5)
        elif arr_geometry == 'chime3':
            arr_radius, mic_num = None, 6
            if arr_rotate_lims == 'auto':
                arr_rotate_lims = ((0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi))
            if spk_arr_dist == 'auto':
                spk_arr_dist = (0.3, 0.5)
            if arr_room_dist == 'auto':
                arr_room_dist = (2.0, 0.5)
        elif arr_geometry == 'libricss':
            arr_radius = (0.0425, 0.0425)
            mic_num = 7
            if spk_arr_dist == 'auto':
                spk_arr_dist = (0.5, 4.5)
            if arr_room_dist == 'auto':
                arr_room_dist = (1.0, 0.5)
        if arr_rotate_lims == 'auto':
            arr_rotate_lims = (None, None, (0, 2 * np.pi))  # rotate by z-axis only
        assert arr_room_dist != 'auto', "configure this parameter"

        if trajectory is not None and trajectory[0].startswith('4points+sin'):
            if arr_radius is None:
                spk_arr_dist = [0.2, 0.2]
            else:
                spk_arr_dist = [max(arr_radius)] * 2 if spk_arr_dist == 'random' else [min(spk_arr_dist)] * 2

        if wall_abs_weights_lims == 'auto':
            wall_abs_weights_lims = [(0.5, 1.0)] * 6
        elif wall_abs_weights_lims is None:
            wall_abs_weights_lims = [(1.0, 1.0)] * 6
        else:
            assert len(wall_abs_weights_lims) == 6, "you should give the weights of six walls"
        seed = np.random.randint(0, 999999999) if seed is None else seed
        args = locals().copy()  # capture the parameters passed to this function or their edited values
        print('Args:')
        print(dict(args), '\n')

        rir_num = sum(rir_nums)
        print('generating rir cfgs. ', end=' ')
        import time
        ts = time.time()

        # to debug, uncommet the following lines
        # new_args = args.copy()
        # del new_args['index']
        # for i in range(rir_num):
        #     generate_rir_cfg_list(i, **new_args)

        # speed up
        with mp.Pool(processes=mp.cpu_count() // 2) as p:
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

    # extract parameters
    if isinstance(mic_num, Tuple):
        assert len(mic_num) == 2, mic_num
        mic_num, sel_mic_num = mic_num
    else:
        mic_num, sel_mic_num = mic_num, mic_num
    if not isinstance(arr_geometry, str) and isinstance(arr_geometry, Iterable):
        assert isinstance(mic_num, Iterable), 'mic_num should be Iterable when arr_geometry is Iterable'
        # choose an array if arr_geometry is a list
        arr_idx = np.random.randint(low=0, high=len(arr_geometry))
        arr_geometry = arr_geometry[arr_idx]
        # choose the radius and mic_num at the same time
        mic_num = mic_num[arr_idx]
        sel_mic_num = sel_mic_num if isinstance(sel_mic_num, int) else sel_mic_num[arr_idx]
        if arr_radius is not None and isinstance(arr_radius[0], Iterable):
            arr_radius = arr_radius[arr_idx]

    # sample radius / RT60 / room_sz / abs_weights
    array_r_this = uniform(*arr_radius) if arr_radius is not None else None
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
    while mic_center is None or (mic_center[:2] < arr_room_dist[1]).any() or (mic_center[:2] > np.array(room_sz)[:2] - arr_room_dist[1]).any():
        mic_center = np.array([
            uniform(room_sz[0] / 2 - arr_room_dist[0], room_sz[0] / 2 + arr_room_dist[0]),
            uniform(room_sz[1] / 2 - arr_room_dist[0], room_sz[1] / 2 + arr_room_dist[0]),
            uniform(*mic_zlim),
        ])

    # generate geometry
    if arr_geometry == 'circular':
        pos_rcv = circular_array_geometry(radius=array_r_this, mic_num=mic_num)
    elif arr_geometry == 'circular+cm':
        pos_rcv = circular_cm_array_geometry(radius=array_r_this, mic_num=mic_num)
    elif arr_geometry == 'linear':
        pos_rcv = linear_array_geometry(radius=array_r_this, mic_num=mic_num)
    elif arr_geometry == 'chime3':
        pos_rcv = chime3_array_geometry()
    elif arr_geometry == 'audiowu_high':
        pos_rcv = audiowu_high_array_geometry()
    elif arr_geometry == 'audiowu_low':
        pos_rcv = audiowu_low_array_geometry()
    else:
        assert arr_geometry == 'libricss', arr_geometry
        pos_rcv = libricss_array_geometry()

    # randomly select channels
    if mic_num != sel_mic_num:
        assert sel_mic_num <= mic_num, mic_num
        sel_chns = np.random.choice(list(range(mic_num)), size=sel_mic_num, replace=False)
        pos_rcv = pos_rcv[sel_chns, :]
    else:
        sel_chns = list(range(mic_num))

    # rotate the array by x/y/z axis
    x_angle, y_angle, z_angle = [None if lim is None else uniform(*lim) for lim in arr_rotate_lims]
    pos_rcv = rotate(pos_rcv=pos_rcv, x_angle=x_angle, y_angle=y_angle, z_angle=z_angle)

    # move center from origin to mic_center
    pos_rcv += mic_center[np.newaxis, :]

    # add small position variations to the (x,y,z) of each mic for simulating position's inperfection
    if mic_pos_var > 0:
        pos_rcv = pos_rcv + uniform(low=-mic_pos_var, high=mic_pos_var, size=pos_rcv.shape)

    # sample speaker postions
    pos_src = []
    # all speaker's loc are randomly sampled
    if trajectory is None:
        for iiii in range(0, spk_num):
            pos_src_i = uniform(0.5, room_sz[0] - 0.5), uniform(0.5, room_sz[1] - 0.5), uniform(spk_zlim[0], spk_zlim[1])
            while spk_arr_dist != 'random' and (norm(pos_src_i - mic_center) < spk_arr_dist[0] or norm(pos_src_i - mic_center) > spk_arr_dist[1]):
                # if the spk_mic_dis requirements are not satisfied, then resample a position
                pos_src_i = uniform(0.5, room_sz[0] - 0.5), uniform(0.5, room_sz[1] - 0.5), uniform(spk_zlim[0], spk_zlim[1])
            pos_src.append(pos_src_i)
        pos_src = np.array(pos_src, dtype=np.float16)
    elif trajectory[0] in ['4points+sin', '4points+sin+eqdist']:
        equal_dist = trajectory[0] == '4points+sin+eqdist'
        spk_arr_dist = np.array(spk_arr_dist + [0])
        for iiii in range(0, spk_num):
            while True:
                traj_i, points4 = generate_4points_sin_trajectory(
                    room_sz=room_sz,
                    rcv_pos=pos_rcv,
                    min_src_array_dist=spk_arr_dist,
                    min_src_boundary_dist=np.array([0.5, 0.5, 0]),
                    src_z=uniform(spk_zlim[0], spk_zlim[1]),
                    desired_dist_pts=trajectory[1],  # the desired distance (10cm) between neighbouring points
                    equal_dist=equal_dist,
                )
                if traj_i is not None:
                    break
            pos_src.append(traj_i.astype(np.float16))
    else:
        raise Exception('Unknown trajectory type: ' + str(trajectory))
    # generate the positions of noise sources
    pos_noise = []
    for iiii in range(noise_num):
        pos_noise.append([uniform(0.1, sz - 0.1) for sz in room_sz])
    pos_noise = np.array(pos_noise)

    # plot room
    # print(x_angle / np.pi * 180, y_angle / np.pi * 180, z_angle / np.pi * 180)
    # plot_room(room_sz=room_sz, pos_src=pos_src, pos_rcv=pos_rcv, pos_noise=pos_noise, saveto=None)
    # plot_room_2d(room_sz=room_sz, pos_src=pos_src, pos_rcv=pos_rcv, pos_noise=pos_noise, saveto=None)

    par = {
        'index': index,
        'RT60': RT60,
        'arr_geometry': f"{arr_geometry}({mic_num},{array_r_this})",
        'selected_channels': sel_chns,
        'room_sz': room_sz,
        'pos_src': pos_src,
        'pos_rcv': pos_rcv.astype(np.float32),
        'pos_noise': pos_noise.astype(np.float32),
        # 'abs_weights': abs_weights,
        'beta': beta,
    }
    return par


def __gen__(par, fs, use_gpu, train_rir_num, val_rir_num, rir_dir, attn_diff_speech, attn_max, attn_diff_noise, split_trajectory):
    index = par['index']
    RT60 = par['RT60']
    room_sz = par['room_sz']
    pos_src = par['pos_src']
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
    gen_rir_func = generate_rir_gpu if use_gpu else generate_rir_cpu
    kwargs_speech, kwargs_noise = ({'att_diff': attn_diff_speech, 'att_max': attn_max}, {'att_diff': attn_diff_noise, 'att_max': attn_max}) if use_gpu else ({}, {})
    if isinstance(pos_src, np.ndarray):
        rir = gen_rir_func(room_sz, pos_src, pos_rcv, RT60, fs, beta=beta, **kwargs_speech)  # reverbrant rir
        rir_dp = gen_rir_func(room_sz, pos_src, pos_rcv, 0, fs)  # direct path rir
    else:
        assert isinstance(pos_src, list), type(pos_src)
        rir, rir_dp = [], []
        for i in range(len(pos_src)):
            if split_trajectory is None:
                rir_i = gen_rir_func(room_sz, pos_src[i], pos_rcv, RT60, fs, beta=beta, **kwargs_speech)  # reverbrant rir
                rir_dp_i = gen_rir_func(room_sz, pos_src[i], pos_rcv, 0, fs)  # direct path rir
            else:
                # split trajectory for solving out of memory issue
                assert split_trajectory > 0, split_trajectory
                rir_i, rir_dp_i = [], []
                pos_src_list = np.array_split(pos_src[i], np.ceil(len(pos_src[i]) / split_trajectory), axis=0)
                for pos_src_i in pos_src_list:
                    rir_i_i = gen_rir_func(room_sz, pos_src_i, pos_rcv, RT60, fs, beta=beta, **kwargs_speech)  # reverbrant rir
                    rir_dp_i_i = gen_rir_func(room_sz, pos_src_i, pos_rcv, 0, fs)  # direct path rir
                    rir_i.append(rir_i_i), rir_dp_i.append(rir_dp_i_i)
                rir_i = np.concatenate(rir_i, axis=0)
                rir_dp_i = np.concatenate(rir_dp_i, axis=0)

            np.save(os.path.join(rir_dir, setdir, str(index) + f'_rir_{i}.npy'), arr=rir_i.astype(np.float16))
            np.savez_compressed(os.path.join(rir_dir, setdir, str(index) + f'_rir_dp_{i}.npz'), arr=rir_dp_i.astype(np.float16))
            rir.append(str(index) + f'_rir_{i}.npy'), rir_dp.append(str(index) + f'_rir_dp_{i}.npz')
        pos_src = np.array(pos_src, dtype=object)  # 每个rir可能不一样长，所以存成object的array
    rir_noise = gen_rir_func(room_sz, pos_noise, pos_rcv, RT60, fs, beta=beta, **kwargs_noise)  # noise rir, uses diffuse model after 20 dB attenuation
    if rir_noise is not None:
        rir_noise = rir_noise.astype(np.float16)
    np.savez(
        save_to,
        fs=fs,
        RT60=RT60,
        room_sz=room_sz,
        pos_src=pos_src,
        pos_rcv=pos_rcv,
        pos_noise=pos_noise,
        rir=rir,
        rir_dp=rir_dp,
        rir_noise=rir_noise,
        arr_geometry=par['arr_geometry'],
        selected_channels=par['selected_channels'],
        beta=beta,
    )


def generate_rir_files(
    rir_cfg: Dict[str, Any],
    rir_dir: str,
    rir_nums: Tuple[int, int, int],
    use_gpu: bool,
    split_trajectory: Optional[int] = None,
    gpus: List[int] = [0],
):
    train_rir_num, val_rir_num, test_rir_num = rir_nums

    pars = rir_cfg['rir_pars']
    fs = rir_cfg['args'].item()['fs']
    attn_diff = rir_cfg['args'].item()['attn_diff']
    attn_diff_noise = None
    if isinstance(attn_diff, tuple):
        attn_diff_speech = attn_diff[0]
        attn_diff_noise = attn_diff[1]
        attn_max = attn_diff[2] if len(attn_diff) >= 3 else 60.0
    else:
        attn_diff_speech = attn_diff  # for old config

    rir_dir = os.path.expanduser(rir_dir)
    if (Path(rir_dir) / 'train').exists() or (Path(rir_dir) / 'validation').exists() or (Path(rir_dir) / 'test').exists():
        ans = input("dir " + rir_dir + ' exists, still generate rir?(y/n)')
        assert ans in ['y', 'n'], ans
        if ans == 'n':
            return
    else:
        os.makedirs(rir_dir, exist_ok=True)

    for setdir in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(rir_dir, setdir), exist_ok=True)

    if not use_gpu:
        from p_tqdm import p_map
        p_map(
            partial(__gen__, fs=fs, use_gpu=use_gpu),
            pars,
            num_cpus=mp.cpu_count() // 2,
        )
    else:
        pbar = tqdm.tqdm(total=len(pars))
        pbar.set_description('generating rirs')
        # use multi-gpu for parallel generation
        def init_env_var(gpus: List[int]):
            i = queue.get()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
            import gpuRIR  # occupy this gpu

        queue = mp.Queue()
        for gid in gpus:
            queue.put(gid)

        p = mp.Pool(processes=len(gpus), initializer=init_env_var, initargs=(queue,))
        for _ in p.imap_unordered(
                partial(
                    __gen__,
                    fs=fs,
                    use_gpu=use_gpu,
                    train_rir_num=train_rir_num,
                    val_rir_num=val_rir_num,
                    rir_dir=rir_dir,
                    attn_diff_speech=attn_diff_speech,
                    attn_max=attn_max,
                    attn_diff_noise=attn_diff_noise,
                    split_trajectory=split_trajectory,
                ),
                pars,
                chunksize=30,
        ):
            pbar.update()
        p.close()
        p.join()


if __name__ == '__main__':
    # python generate_rirs.py --help
    # examples:
    # python generate_rirs.py --spk_num=2 --room_size_lims="[[4,10],[4,10],[3,4]]" --mic_zlim="[1.4,1.6]" --spk_zlim="[1.3,1.8]" --RT60_lim=[0.1,1.0] --rir_nums=[20000,2000,2000] --arr_geometry=chime3 --arr_radius=null --mic_num=6 --spk_arr_dist=[0.5,0.5] --arr_room_center_dist=0.5 --attn_diff=[15.0,15.0,40.0] --fs=8000 --save_to=~/datasets/CHiME3_moving_rirs/rir_cfg.npz --rir_dir=~/datasets/CHiME3_moving_rirs --trajectory=['4points+sin',0.05]
    parser = ArgumentParser(description='code for generating RIRs by Changsheng Quan @ Westlake University')
    parser.add_function_arguments(generate_rir_cfg_list)  # add_argument for the function generate_rir_cfg_list
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--gpus', type=List[int], default=[0], help='the gpus used for simulation')
    parser.add_argument('--split_trajectory', type=Optional[int], default=None, help='set this parameter to a small value if out-of-memory')

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
        gpus=args.gpus,
        split_trajectory=args.split_trajectory,
    )
