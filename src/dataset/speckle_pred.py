# from src.config import RAW_DATA_DIR
import numpy as np

from src.dataset.process import collected_signal, simulated_signal, target_image


def speckle_pred_inv(path_x, path_y, select="white", rand_select="both"):
    Y_random, _ = collected_signal(path=path_y, select=select, rand_select=rand_select)
    X_random, _ = target_image(path=path_x, select=select, rand_select=rand_select)
    X_pinv = np.linalg.pinv(X_random)
    S = np.dot(X_pinv, Y_random)
    return S


def speckle_pred_simulate(path_x, Y_all, select, rand_select):
    Y_random, _ = simulated_signal(Y_all, select=select, rand_select=rand_select)
    X_random, _ = target_image(path=path_x, select=select, rand_select=rand_select)
    X_pinv = np.linalg.pinv(X_random)
    S = np.dot(X_pinv, Y_random)
    return S.T
