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


def load_npz_signal(file_name):
    """
    params:
    file: collected data path

    return:
    data_skipped(6020, 2500)
    """
    signal = load_npz_file(file_name=file_name)
    signal = signal[:, ::2]
    return signal


# 1500+10+1500のとき
def target_image(path, select="both", rand_select="both"):
    """
    params:
    path: 対象画像のpath
    select:str :
        both: 差分を取った画像を出力
        white: 白文字画像出力
        black: 黒文字画像出力
    """
    data_x = load_npz_file(file_name=path)
    random_0_3000 = data_x[:3000, :]
    mnist = data_x[3000:3020, :]
    random_3020_6020 = data_x[3020:, :]
    mnist_white = mnist[::2, :]
    mnist_black = mnist[1::2, :]
    random = np.vstack((random_0_3000, random_3020_6020))
    random_white = random[::2, :]
    random_black = random[1::2, :]
    if select == "both":
        X_mnist = mnist_white - mnist_black
    elif select == "white":
        X_mnist = mnist_white
    elif select == "black":
        X_mnist = mnist_black
    else:
        ValueError("arg:select is wrong!")

    if rand_select == "both":
        X_random = random_white - random_black
    elif rand_select == "white":
        X_random = random_white
    elif rand_select == "black":
        X_random = random_black
    else:
        ValueError("arg:select is wrong!")
    return X_random, X_mnist


# 1500+10+1500のとき
def collected_signal(path, select="both", rand_select="both"):
    """
    params:
    path: 対象画像のpath
    select:str :
        both: 差分を取った画像を出力
        white: 白文字画像出力
        black: 黒文字画像出力
    """
    signal = load_npz_signal(file_name=path)
    random_0_3000 = signal[:3000, :]
    mnist = signal[3000:3020, :]
    random_3020_6020 = signal[3020:, :]
    mnist_white = mnist[::2, :]
    mnist_black = mnist[1::2, :]
    random = np.vstack((random_0_3000, random_3020_6020))
    random_white = random[::2, :]
    random_black = random[1::2, :]
    if select == "both":
        Y_mnist = mnist_white - mnist_black
    elif select == "white":
        Y_mnist = mnist_white
    elif select == "black":
        Y_mnist = mnist_black
    else:
        ValueError("arg:select is wrong!")

    if rand_select == "both":
        Y_random = random_white - random_black
    elif rand_select == "white":
        Y_random = random_white
    elif rand_select == "black":
        Y_random = random_black
    else:
        ValueError("arg:select is wrong!")
    return Y_random, Y_mnist


def simulated_signal(signal, select="both", rand_select="both"):
    """
    params:
    signal: X, S_.T の積(6020, 784)
    select:str :
        both: 差分を取った画像を出力
        white: 白文字画像出力
        black: 黒文字画像出力
    """
    random_0_3000 = signal[:3000, :]
    mnist = signal[3000:3020, :]
    random_3020_6020 = signal[3020:, :]
    mnist_white = mnist[::2, :]
    mnist_black = mnist[1::2, :]
    random = np.vstack((random_0_3000, random_3020_6020))
    random_white = random[::2, :]
    random_black = random[1::2, :]
    if select == "both":
        Y_mnist = mnist_white - mnist_black
    elif select == "white":
        Y_mnist = mnist_white
    elif select == "black":
        Y_mnist = mnist_black
    else:
        ValueError("arg:select is wrong!")

    if rand_select == "both":
        Y_random = random_white - random_black
    elif rand_select == "white":
        Y_random = random_white
    elif rand_select == "black":
        Y_random = random_black
    else:
        ValueError("arg:select is wrong!")
    return Y_random, Y_mnist
