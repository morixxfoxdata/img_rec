import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)

def min_max_normalize(input_tensor):
    """
    入力テンソルに対してmin-max正規化
    データの最小値を0、最大値を1にスケーリング

    * 注釈:
      - **min-max正規化**: データの最小値と最大値を用い、各値から最小値を引いた後に、(最大値-最小値)で割ることで0〜1の範囲に変換する手法です。
    """
    min_val = input_tensor.min()
    max_val = input_tensor.max()
    
    # ゼロ除算回避のため、最小値と最大値が等しい場合は、全ての要素を0にします。
    if max_val == min_val:
        return torch.zeros_like(input_tensor)
    
    normalized_tensor = (input_tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def min_max_normalize_np(input):
    """
    入力テンソルに対してmin-max正規化
    データの最小値を0、最大値を1にスケーリング

    * 注釈:
      - **min-max正規化**: データの最小値と最大値を用い、各値から最小値を引いた後に、(最大値-最小値)で割ることで0〜1の範囲に変換する手法です。
    """
    min_val = input.min()
    max_val = input.max()
    
    # ゼロ除算回避のため、最小値と最大値が等しい場合は、全ての要素を0にします。
    if max_val == min_val:
        return np.zeros_like(input)
    
    normalized_tensor = (input - min_val) / (max_val - min_val)
    return normalized_tensor

def standardize(input):
    mean = input.mean()
    std = input.std()
    epsilon = 1e-7
    normalized_tensor = (input - mean) / (std + epsilon)
    print("mean (before):", mean.item())
    print("std (before):", std.item())
    print("mean (after):", normalized_tensor.mean().item())
    print("std (after):", normalized_tensor.std().item())
    return normalized_tensor

def total_variation_loss(x):
    # 縦方向の差分 |x[i, j+1] - x[i, j]|
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

    # 横方向の差分 |x[i+1, j] - x[i, j]|
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

    return torch.sum(tv_h) + torch.sum(tv_w)


def image_save(x, y, epoch, num, select, rand_select, model, lr, tv, scale, kernel_size):
    """
    x: 正解画像をFlatten(784次元)した配列
    y: 再構成画像をFlatten(784次元)した配列
    save_path: 保存ファイルパス (デフォルト: reconstruction_result.png)
    """
    save_dir = os.path.join("results", "pix28", 
                            f"m_{select}+r_{rand_select}", str(model), 
                            f"lr{lr}_tv{tv}_scale{scale}_kernel{kernel_size}")
    if not os.path.exists(save_dir):  # 存在しなければ作る
        os.makedirs(save_dir)
    img_path=f"num{num}_ep{epoch}.png"


    # 28x28にreshapeして可視化できる形にする
    x_img = x.reshape(28, 28)
    y_img = y.reshape(28, 28)

    # MSE, SSIM, PSNRを計算
    mse_val = mean_squared_error(x_img, y_img)
    ssim_val = ssim(x_img, y_img, data_range=x_img.max() - x_img.min())
    psnr_val = psnr(x_img, y_img, data_range=x_img.max() - x_img.min())

    # 2つの画像を並べて表示するための設定
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    if select=="both":
        axes[0].imshow(x_img, cmap="gray", vmin=-1, vmax=1)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(y_img, cmap="gray", vmin=-1, vmax=1)
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")
    else:
        axes[0].imshow(x_img, cmap="gray")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(y_img, cmap="gray")
        axes[1].set_title("Reconstructed")
        axes[1].axis("off")
    # SSIM, MSE, PSNR を小数点以下5桁までテキストにまとめる
    text_str = (
        f"SSIM: {ssim_val:.5f}\n"
        f"MSE : {mse_val:.5f}\n"
        f"PSNR: {psnr_val:.5f}"
    )

    # 2枚目のサブプロット領域(axes[1])にテキストを配置
    # (X座標=0.5, Y座標=-0.1 は、サブプロット座標系の外側・下部)
    axes[1].text(
        0.5, -0.1, text_str,
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=10
    )

    plt.tight_layout()
    save_file = os.path.join(
        save_dir, img_path
    )
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()
    return mse_val, ssim_val, psnr_val