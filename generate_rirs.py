# -*- coding: utf-8 -*-
# GENERATE_RIR_LIST generate all the possible permutations according the
# angles assigned to tr, cv, and tt
#
# allocate can be an array like [10 3] or [9:2:2]: if it has a length
# of two, it means tr & cv share the same rir set;
# rir_nums gives the num of rirs of tr, cv, and tt, thus it can be [20000
# 5000 3000]

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
# import cupy as cp
from typing import List, Tuple, Union
import numpy as np
from scipy.signal import convolve, resample
import argparse
import gpuRIR
from numpy.random import uniform
from numpy.linalg import norm
import tqdm
import json

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)


def is_valid_beta(beta: Union[List[float], np.ndarray]) -> bool:
    return not np.isclose(beta, 0).any()


def estimate_minimal_RT60(room_sz: Union[List[float], np.ndarray]) -> float:
    V = 1.0
    for v in room_sz:
        V = V * v
    S = (room_sz[0] * room_sz[1] + room_sz[0] * room_sz[2] + room_sz[1] * room_sz[2]) * 2
    T60 = 0.161 * V / S
    return T60


def is_valid_RT60_for_room(room_sz: Union[List[float], np.ndarray], RT60: float, eps: float = 1e-4) -> bool:
    RT60m = estimate_minimal_RT60(room_sz)
    if RT60 < RT60m + eps:
        return False
    else:
        return True


def generate_rir(room_sz: Union[List[float], np.ndarray],
                 pos_src: Union[List[List[float]], np.ndarray],
                 pos_rcv: Union[List[List[float]], np.ndarray],
                 RT60: float,
                 fs: int,
                 mic_pattern: str,
                 abs_weights: Union[List[float], np.ndarray] = [1.0] * 6,
                 att_diff: float = None) -> np.ndarray:
    """使用gpuRIR生成rir

    Args:
        room_sz: 房间尺寸
        pos_src: 声源位置
        pos_rcv: 麦克风位置
        RT60: 房间RT60
        fs: 采样率
        mic_pattern: 麦克风的类型。可选值有{"omni", "homni", "card", "hypcard", "subcard", "bidir"}
        abs_weight: 房间吸收参数
        att_diff: Attenuation when start using the diffuse reverberation model [dB]. 合适的值可以加速rir的生成

    Returns:
        np.ndarray: 3D ndarray. The first axis is the source, the second the receiver and the third the time.
    """
    Tmax = gpuRIR.att2t_SabineEstimator(60.0, RT60)
    if att_diff != None:
        Tdiff = gpuRIR.att2t_SabineEstimator(att_diff, RT60)  # Time to start the diffuse reverberation model [s]
        nb_img = gpuRIR.t2n(Tdiff, room_sz)  # Number of image sources in each dimension
    else:
        Tdiff = None
        nb_img = gpuRIR.t2n(Tmax, room_sz)

    beta = gpuRIR.beta_SabineEstimation(room_sz, RT60, abs_weights=abs_weights)  # reflection coefficients
    if is_valid_beta(beta) == False:
        import warnings
        warnings.warn(f'beta is invalid for gpuRIR, which means the given RT60={RT60} could not achieved with the given room_sz={room_sz} and abs_weights={abs_weights}')

    if mic_pattern == 'omni':  # fix the bug in simulateRIR, when mic_pattern is loaded from a json file
        mic_pattern = 'omni'

    rir = gpuRIR.simulateRIR(room_sz=room_sz, beta=beta, pos_src=pos_src, pos_rcv=pos_rcv, nb_img=nb_img, Tmax=Tmax, fs=fs, mic_pattern=mic_pattern, Tdiff=Tdiff)

    # import matplotlib.pyplot as plt
    # t = np.arange(rir.shape[2]) / fs
    # plt.plot(t, rir[0, 0, :])
    # plt.show()
    return rir


def normalize(vec: np.ndarray) -> np.ndarray:
    # get unit vector
    vec = vec / norm(vec)
    vec = vec / norm(vec)
    assert np.isclose(norm(vec), 1), 'norm of vec is not close to 1'
    return vec


def plot_room(room_sz: Union[List[float], np.ndarray], pos_src: np.ndarray, pos_rcv: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pos_rcv[:, 0], pos_rcv[:, 1], pos_rcv[:, 2])
    ax.scatter(pos_src[:, 0], pos_src[:, 1], pos_src[:, 2])
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_xlim3d([0, room_sz[0]])
    ax.set_ylim3d([0, room_sz[1]])
    ax.set_zlim3d([0, room_sz[2]])
    plt.show()
    plt.close()


def find_random_vertical_vectors(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """find two vectors, where the two vectors and vec are vertical to each other

    Args:
        vec (array like): a vector

    Returns:
        two vectors: array like
    """

    # find the ax where its value not zero
    index = 0
    for idx, v in enumerate(vec):
        if v != 0:
            index = idx
    # A是向量的垂线
    A = [[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]]
    del A[index]
    v1 = np.array(A[0])
    v1 = normalize(v1)
    # v1和vec的垂向量
    v1v = np.cross(v1, vec)
    v1v = normalize(v1v)
    assert np.isclose(np.dot(v1, v1v), 0), 'v1v is not vertical to v1'
    assert np.isclose(np.dot(v1, vec), 0), 'v1 is not vertical to vec'
    assert np.isclose(np.dot(v1v, vec), 0), 'v1v is not vertical to vec'

    # 将v1随机旋转theta弧度，得到v1t和v1tv，其中v1t为第一个mic的位置
    theta = uniform(-0.01, 0.01) * np.pi * 2  # +/- 3.6度
    v1t = v1 * np.cos(theta) + v1v * np.sin(theta)
    v1tv = np.cross(v1t, vec)
    v1t, v1tv = normalize(v1t), normalize(v1tv)
    assert np.isclose(np.dot(v1t, v1tv), 0), 'v1tv is not vertical to v1t'
    assert np.isclose(np.dot(v1t, vec), 0), 'v1t is not vertical to vec'
    assert np.isclose(np.dot(v1tv, vec), 0), 'v1tv is not vertical to vec'
    assert np.isclose(np.dot(v1, v1t) / (norm(v1) * norm(v1t)), np.cos(theta)), "angle error"
    return v1t, v1tv


def generate_rir_cfg_list(spk_num=2,
                          xlim=[3, 8],
                          ylim=[3, 8],
                          zlim=[3, 4],
                          array_r=0.05,
                          RT60lim=[0.1, 1.0],
                          rir_num=28000,
                          mic_num=8,
                          array_type='circular',
                          mic_pattern='omni',
                          fs=16000,
                          abs_weights=[1.0] * 6,
                          save_to=None):
    """
    draw room length from xlim like [3m, 8m], room width from ylim like [3m, 8m], and room height from zlim like [3m, 4m];
    draw T60 from RT60lim like [0.1, 0.6]s;
    draw spk pos and array pos 0.5m away from the surface of room

    Args:
        rir_num (int, optional): [description]. Defaults to 28000.

    Returns:
        list[dict]: a list of parameters to generate rirs
    """
    if os.path.exists(save_to):
        with open(save_to, 'r', encoding='utf-8') as file:
            cfg = json.load(file)
            if cfg['spk_num'] != spk_num or cfg['mic_num'] != mic_num or cfg['rir_num'] != rir_num or cfg['array_type'] != array_type or cfg['array_r'] != array_r or cfg[
                    'fs'] != fs or cfg['RT60lim'] != RT60lim or cfg['xlim'] != xlim or cfg['ylim'] != ylim or cfg['zlim'] != zlim or cfg['abs_weights'] != abs_weights:
                raise Exception("file " + save_to + ' already exists, but with a different paramters!!! Delete it to re-generate')
            else:
                print('load rir cfgs from file ' + save_to)
                return cfg

    assert array_type == 'circular', "only supports circular array for now"
    assert spk_num == 2, "Only supports 2-speaker cases"

    rir_pars = []
    pbar = tqdm.tqdm(total=rir_num)
    pbar.set_description(f"generating rir cfgs")
    for i in range(rir_num):
        pbar.update()
        RT60 = uniform(*RT60lim)
        room_sz = [uniform(*xlim), uniform(*ylim), uniform(*zlim)]
        while is_valid_RT60_for_room(room_sz, RT60) == False:
            room_sz = [uniform(*xlim), uniform(*ylim), uniform(*zlim)]
            RT60 = uniform(*RT60lim)
            beta = gpuRIR.beta_SabineEstimation(room_sz, RT60, abs_weights=abs_weights)  # reflection coefficients

        # microphone positions
        pos_rcv = np.empty((mic_num, 3))
        mic_center = np.array([uniform(room_sz[0] / 2 - 0.5, room_sz[0] / 2 + 0.5), uniform(room_sz[1] / 2 - 0.5, room_sz[1] / 2 + 0.5), 1.5])
        # 在半径为1的水平圆面内部随机找一点，将该点与原点形成的向量作为圆形麦克风阵列平面的法向量
        norm_vec = np.array([uniform(-1, 1), uniform(-1, 1), 0])
        while np.linalg.norm(norm_vec) == 0:
            norm_vec = np.array([uniform(-1, 1), uniform(-1, 1), 0])
        norm_vec = normalize(norm_vec)  # 单位向量
        # 找到与法向量垂直的两条单位向量
        v1t, v1tv = find_random_vertical_vectors(norm_vec)
        # 将v1t旋转angle角度，来生成mic的位置
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
        for idx, angle in enumerate(angles):
            pos_rcv[idx, :] = normalize(v1t * np.cos(angle) + v1tv * np.sin(angle))
        # 检查是否满足条件
        fai = 2 * np.pi / mic_num
        for j in range(mic_num - 1):
            pos_j = pos_rcv[j, :]
            pos_jp = pos_rcv[j + 1, :]
            assert np.isclose(np.dot(pos_j, pos_jp) / (norm(pos_j) * norm(pos_jp)), np.cos(fai)), "angle error"
            assert np.isclose(np.dot(pos_j, norm_vec), 0), 'pos_j is not vertical to norm_vec'
            assert np.isclose(np.dot(pos_jp, norm_vec), 0), 'pos_jp is not vertical to norm_vec'
            assert np.isclose(norm(pos_j), 1), '|pos_j| != 1'
            assert np.isclose(norm(pos_jp), 1), '|pos_j| != 1'
        pos_rcv = pos_rcv * array_r
        pos_rcv = pos_rcv + mic_center

        # speaker postions
        pos_src = np.empty((spk_num, 3))
        # first speaker's loc is randomly sampled
        pos_src[0, 0] = uniform(0.5, room_sz[0] - 0.5)  # x
        pos_src[0, 1] = uniform(0.5, room_sz[1] - 0.5)  # y
        pos_src[0, 2] = 1.5  # z
        # the second speaker's loc is sampled according to a uniformly sampled angle to spk 1 and mic center
        theta = uniform(0, 1) * np.pi * 2  # 0~360度
        mc2s1 = pos_src[0, :] - mic_center  # mic center to speaker 1
        z = np.array([0, 0, 1])
        mc2s2 = mc2s1 * np.cos(theta) + (1 - np.cos(theta)) * np.dot(mc2s1, z) * z + np.sin(theta) * np.cross(z, mc2s1)  # mic center to speaker 2

        # 将mc2s2的x,y缩放到x_lims,y_lims构成的正方形区域内部
        x, y = mc2s2[0], mc2s2[1]
        x_lims = [0.5 - mic_center[0], room_sz[0] - 0.5 - mic_center[0]]
        y_lims = [0.5 - mic_center[1], room_sz[1] - 0.5 - mic_center[1]]

        # 第一步缩放到x边界上面去
        if x >= 0:  # 缩放到x正边界
            scale = abs(1 / x * x_lims[1])
            y = y * scale
            x = x * scale  # x=x*abs(1/x*x_lims[1])=x/scale=x_lims[1]
        else:
            scale = abs(1 / x * x_lims[0])
            y = y * scale
            x = x * scale
        assert np.allclose(np.array([x, y]), mc2s2[:2] * scale)
        # 第二步，检查y是否越界。
        if y >= 0 and y > y_lims[1]:  # 缩放到y正边界
            scale = abs(1 / y * y_lims[1])
            y = y * scale
            x = x * scale
        elif y < 0 and y < y_lims[0]:  # 缩放到y负边界
            scale = abs(1 / y * y_lims[0])
            y = y * scale
            x = x * scale

        rescale = uniform(0, 1)
        mc2s2 = np.array([x * rescale, y * rescale, 0])
        pos_src[1, :] = mc2s2 + mic_center
        # check loc in range
        assert (pos_src[1, :2] >= 0.5).all()
        assert (pos_src[1, :2] <= (np.array(room_sz[:2]) - 0.5)).all()
        # check theta is right
        theta_real = np.arccos(np.dot(mc2s2, mc2s1) / np.linalg.norm(mc2s1) / np.linalg.norm(mc2s2))
        assert np.allclose(theta_real, theta) or np.allclose(theta_real + theta, np.pi * 2)
        # plot_room(room_sz, pos_src, pos_rcv)
        angle = theta if theta <= np.pi else (theta - np.pi)
        angle = angle / np.pi * 180
        par = {'file': str(i) + '.npz', 'RT60': RT60, 'room_sz': room_sz, 'pos_src': pos_src.tolist(), 'pos_rcv': pos_rcv.tolist(), 'angle': angle}
        rir_pars.append(par)

    cfg = {
        'spk_num': spk_num,
        'mic_num': mic_num,
        'rir_num': rir_num,
        'array_type': array_type,
        'array_r': array_r,
        'mic_pattern': mic_pattern,
        'fs': fs,
        'RT60lim': RT60lim,
        'xlim': xlim,
        'ylim': ylim,
        'zlim': zlim,
        'abs_weights': abs_weights,
        'rir_pars': rir_pars
    }

    if save_to != None:
        dir = os.path.dirname(save_to)
        if len(dir) > 0:
            os.makedirs(dir, exist_ok=True)
        with open(save_to, 'w', encoding='utf-8') as file:
            json.dump(cfg, file, sort_keys=False, indent=4, separators=(',', ':'))
            file.close()
    # import pandas as pd
    # df = pd.DataFrame(rir_pars)
    # df.plot.scatter(x='RT60',y='angle')
    return cfg


def generate_rir_files(rir_cfg, rir_dir, train_num, validation_num, test_num):
    pars = rir_cfg['rir_pars']
    fs = rir_cfg['fs']
    abs_weights = rir_cfg['abs_weights']
    mic_pattern = rir_cfg['mic_pattern']

    if os.path.exists(rir_dir):
        print("exist dir " + rir_dir + ', so not generate rir')
        return
    else:
        os.makedirs(rir_dir, exist_ok=True)

    pbar = tqdm.tqdm(total=len(pars))
    pbar.set_description('generating rirs')
    for i, par in enumerate(pars):
        pbar.update()

        file = par['file']
        RT60 = par['RT60']
        room_sz = par['room_sz']
        pos_src = np.array(par['pos_src'])
        pos_rcv = np.array(par['pos_rcv'])

        if i < train_num:
            setdir = 'train'
        elif i >= train_num and i < train_num + validation_num:
            setdir = 'validation'
        else:
            setdir = 'test'
        os.makedirs(os.path.join(rir_dir, setdir), exist_ok=True)

        rir = generate_rir(room_sz, pos_src, pos_rcv, RT60, fs, mic_pattern, abs_weights)
        np.savez_compressed(os.path.join(rir_dir, setdir, file), sr=fs, RT60=RT60, abs_weights=abs_weights, room_sz=room_sz, pos_src=pos_src, pos_rcv=pos_rcv, speech_rir=rir)


def load_and_resample_impulse_response(rir_file_path: str, resample_fs: int) -> np.ndarray:
    """Load and resample RIR to resample_fs

    Args:
        rir_file_path: npz file path of the rir
        resample_fs: resample rir to resample_fs

    Returns:
        np.ndarray: rir
    """
    rir_all = np.load(rir_file_path)
    rir, fs = rir_all['rir'], rir_all['fs']
    if resample_fs == fs:
        return rir

    re_len = int(rir.shape[2] * resample_fs / fs)
    rir_r = resample(rir, re_len, axis=2)
    return rir_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NBSS')

    parser.add_argument('--rir_dir', default='dataset/rir_cfg_4', type=str, help='the dir of rirs')
    parser.add_argument('--rir_cfg_file', default='configs/rir_cfg_4.json', type=str, help='rir cfg file')

    parser.add_argument('--train_num', default=20000, type=int, help='num of rirs for train set')
    parser.add_argument('--validation_num', default=5000, type=int, help='num of rirs for validation set')
    parser.add_argument('--test_num', default=3000, type=int, help='num of rirs for test set')

    parser.add_argument('--spk', default=2, type=int, help='num of speakers (default: 2)')
    parser.add_argument('--mic', default=8, type=int, help='num of mics (default: 8)')

    args = parser.parse_args()

    rir_dir = args.rir_dir
    rir_cfg_file = args.rir_cfg_file
    spk = args.spk
    mic = args.mic

    train_num = args.train_num
    validation_num = args.validation_num
    test_num = args.test_num

    rir_cfg = generate_rir_cfg_list(mic_num=mic, rir_num=train_num + validation_num + test_num, save_to=rir_cfg_file)
    generate_rir_files(rir_cfg, rir_dir, train_num=train_num, validation_num=validation_num, test_num=test_num)
