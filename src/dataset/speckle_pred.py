# from src.config import RAW_DATA_DIR
import numpy as np
from src.dataset.process import load_npz_file, load_npz_signal, target_image, collected_signal

def speckle_pred_inv(path_x, path_y, select="white"):
    Y_random, _ = collected_signal(path=path_y, select=select)
    X_random, _ = target_image(path=path_x, select=select)
    X_pinv = np.linalg.pinv(X_random)
    S = np.dot(X_pinv, Y_random)
    return S