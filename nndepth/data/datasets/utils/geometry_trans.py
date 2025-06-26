import numpy as np
from scipy.spatial.transform import Rotation as R


def pos_quat2SE(quat_data):
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3, 0:3] = np.matrix(SO)
    SE[0:3, 3] = np.matrix(quat_data[0:3]).T
    return SE
