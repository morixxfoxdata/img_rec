import os
import numpy as np

from src.config import RAW_DATA_DIR
# python -m src.dataset.process
# file_name = "Rand+Mnist+Rand_pix28x28_image(1500+10+1500)x2_sig2500_ffts02_DMD200x200_1547.npz"

def load_npz_file(file_name):
    """
    data/raw 下にある npz ファイルを読み込む関数。
    file_name は "Rand+Mnist+Rand_pix28x28_image(1500+10+1500)x2_sig2500_ffts02_DMD200x200_1547.npz" など、ファイル名のみを想定。
    return: ndarray
    """
    # RAW_DATA_DIR とファイル名を結合して絶対パスを作成
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    
    # npzファイルを読み込み
    data = np.load(file_path)["arr_0"]
    
    return data
