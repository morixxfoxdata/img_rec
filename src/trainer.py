import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset.process import load_npz_file, load_npz_signal, target_image, collected_signal
from src.dataset.speckle_pred import speckle_pred_inv
from src.utils.inv_recon import img_reconstruction
from src.utils.utils import np_to_torch, min_max_normalize, total_variation_loss, image_save, standardize
from src.models.linear import FCModel
from src.models.GIDC import GIDC28
# python -m src.trainer
file_y = "Rand+Mnist+Rand_pix28x28_image(1500+10+1500)x2_sig2500x4wave.npz"
file_x = "Rand+Mnist+Rand_size28x28_image(1500+10+1500)x2.npz"





def train_simple(collected_path, target_path, select):
# =============================================
    num_epochs = 1000
    lr = 0.001
    TV_strength = 0
# =============================================
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)
    Y_random, Y_mnist = collected_signal(path=collected_path, select=select)
    X_random, X_mnist = target_image(path=target_path, select=select)
    print("======================================")
    print("Y_mnist shape:", Y_mnist.shape)
    print("Y_random shape:", Y_random.shape)
    print("X_mnist shape:", X_mnist.shape)
    print("X_random shape:", X_random.shape)
    print("X_mnist min, max:", X_mnist.min(), X_mnist.max())
    print("X_random min, max:", X_random.min(), X_random.max())
    print("======================================")
    print("ランダムパターンからspeckle_patternsを推定します。pinvを利用します。")
    S = 2 * speckle_pred_inv(path_x=target_path, path_y=collected_path, select=select)
    print("speckle by random:", S.min(), S.max(), S.shape)
    print("======================================")
    S_tensor = np_to_torch(S).float().to(device)
    Y_mnist_tensor = np_to_torch(Y_mnist).float()
    criterion = nn.MSELoss()
    recon_list = []
    for num in range(10):
        print(f"\n================ Image {num} の学習開始 ================\n")
        y_ = Y_mnist_tensor[num].to(device)
        print(y_.shape)
        model = FCModel(input_size=10000, hidden_size=1024, output_size=784, select=select).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(y_).reshape((1, 1, 28, 28))
            tv = TV_strength * total_variation_loss(output)
            Y_dash = torch.mm(output.reshape(1, 28**2), S_tensor)
            loss = criterion(Y_dash, y_.unsqueeze(0)) + tv
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 500 == 0:
                print(
                f"Image {num}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}"
                )
                model.eval()
                with torch.no_grad():
                    reconstucted_target = (
                        model(y_).squeeze(0).squeeze(0).reshape(784))
                    # print("再構成画像の shape:", reconstucted_target.shape)
                mse_val, ssim_score, psnr = image_save(
                    x=X_mnist[num, :],
                    y=reconstucted_target.cpu().numpy(),
                    epoch=epoch + 1,
                    num=num,
                    select=select,
                    model=model.model_name,
                    lr=lr,
                    tv=TV_strength)
                print(f"mse:{mse_val:.5f}, ssim:{ssim_score:.5f}, PSNR: {psnr:.5f}")
        print(f"\n================ Image {num} の学習終了 ================\n")
        recon_list.append(reconstucted_target.detach().cpu().numpy())
    print("モデルトレーニングが完了しました。")
    return recon_list



def train_gidc(collected_path, target_path, select):
# =============================================
    num_epochs = 2000
    lr = 0.02
    TV_strength = 1e-8
# =============================================
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)
    Y_random, Y_mnist = collected_signal(path=collected_path, select=select)
    X_random, X_mnist = target_image(path=target_path, select=select)
    Y_mnist = Y_mnist * (-1)
    Y_random = Y_random * (-1)
    print("======================================")
    print("Y_mnist shape:", Y_mnist.shape)
    print("Y_random shape:", Y_random.shape)
    print("X_mnist shape:", X_mnist.shape)
    print("X_random shape:", X_random.shape)
    print("X_mnist min, max:", X_mnist.min(), X_mnist.max())
    print("X_random min, max:", X_random.min(), X_random.max())
    print("======================================")
    print("ランダムパターンからspeckle_patternsを推定します。pinvを利用します。")
    S = 2 * speckle_pred_inv(path_x=target_path, path_y=collected_path, select=select)
    print("speckle by random:", S.min(), S.max(), S.shape)
    X_mnist_first = img_reconstruction(S, Y_mnist)
    print("X_pinv:", X_mnist_first.min(), X_mnist_first.max(), X_mnist_first.shape)
    print("======================================")
    S_tensor = np_to_torch(S).float().to(device)
    Y_mnist_tensor = np_to_torch(Y_mnist).float()
    X_input_tensor = np_to_torch(X_mnist_first).float()
    criterion = nn.MSELoss()
    recon_list = []
    for num in range(10):
        print(f"\n================ Image {num} の学習開始 ================\n")
        # input = min_max_normalize(X_input_tensor[num].reshape((1, 1, 28, 28))).to(device)
        input = standardize(X_input_tensor[num].reshape((1, 1, 28, 28))).to(device)
        y_ = Y_mnist_tensor[num].to(device)
        # print(y_.shape)
        model = GIDC28(kernel_size=5, name="gidc", select=select).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(input).reshape((1, 1, 28, 28))
            tv = TV_strength * total_variation_loss(output)
            Y_dash = torch.mm(output.reshape(1, 28**2), S_tensor)
            loss = criterion(Y_dash, y_.unsqueeze(0)) + tv
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(
                f"Image {num}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}"
                )
                model.eval()
                with torch.no_grad():
                    reconstucted_target = (
                        model(input).squeeze(0).squeeze(0).reshape(784))
                    # print("再構成画像の shape:", reconstucted_target.shape)
                mse_val, ssim_score, psnr = image_save(
                    x=X_mnist[num, :],
                    y=reconstucted_target.cpu().numpy(),
                    epoch=epoch + 1,
                    num=num,
                    select=select,
                    model=model.model_name,
                    lr=lr,
                    tv=TV_strength)
                print(f"mse:{mse_val:.5f}, ssim:{ssim_score:.5f}, PSNR: {psnr:.5f}")
        print(f"\n================ Image {num} の学習終了 ================\n")
        recon_list.append(reconstucted_target.detach().cpu().numpy())
    print("モデルトレーニングが完了しました。")
    return recon_list

