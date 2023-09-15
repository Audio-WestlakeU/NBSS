import numpy as np
import tqdm
from numpy.linalg import norm


def normalize(vec: np.ndarray) -> np.ndarray:
    # get unit vector
    vec = vec / norm(vec)
    vec = vec / norm(vec)
    assert np.isclose(norm(vec), 1), 'norm of vec is not close to 1'
    return vec


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
