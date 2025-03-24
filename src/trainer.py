import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import SPECKLE_DATA_DIR

from .dataset.process import (
    collected_signal,
    load_npz_file,
    simulated_signal,
    target_image,
)
from .dataset.speckle_pred import speckle_pred_inv, speckle_pred_simulate
from .models.GIDC import GIDC28
from .models.linear import FCModel
from .utils.inv_recon import img_reconstruction
from .utils.utils import (
    image_save,
    min_max_normalize,
    np_to_torch,
    standardize,
    total_variation_loss,
)

# python -m src.trainer
file_y = "Rand+Mnist+Rand_pix28x28_image(1500+10+1500)x2_sig2500x4wave.npz"
file_x = "Rand+Mnist+Rand_size28x28_image(1500+10+1500)x2.npz"
file_s = "mask_patterns_length10_core1e-05.npz"
seed = 42
np.random.seed(seed)

# PyTorchのCPU用シードを固定
torch.manual_seed(seed)
# PyTorchのGPU用シードを固定（複数GPUがある場合は全てに対して設定）
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# cuDNNの非決定性を防ぐための設定
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_simple(collected_path, target_path, select, rand_select, scale):
    # =============================================
    num_epochs = 2000
    lr = 0.0002
    TV_strength = 1e-8
    # =============================================
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)
    Y_random, Y_mnist = collected_signal(
        path=collected_path, select=select, rand_select=rand_select
    )
    X_random, X_mnist = target_image(
        path=target_path, select=select, rand_select=rand_select
    )
    print("======================================")
    print("Y_mnist shape:", Y_mnist.shape)
    print("Y_random shape:", Y_random.shape)
    print("X_mnist shape:", X_mnist.shape)
    print("X_random shape:", X_random.shape)
    print("X_mnist min, max:", X_mnist.min(), X_mnist.max())
    print("X_random min, max:", X_random.min(), X_random.max())
    print("======================================")
    print("ランダムパターンからspeckle_patternsを推定します。pinvを利用します。")
    S = scale * speckle_pred_inv(
        path_x=target_path,
        path_y=collected_path,
        select=select,
        rand_select=rand_select,
    )
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
        model = FCModel(
            input_size=10000, hidden_size=1024, output_size=784, select=select
        ).to(device)
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
            if (epoch + 1) % 100 == 0:
                print(
                    f"Image {num}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}"
                )
                model.eval()
                with torch.no_grad():
                    reconstucted_target = model(y_).squeeze(0).squeeze(0).reshape(784)
                    # print("再構成画像の shape:", reconstucted_target.shape)
                mse_val, ssim_score, psnr = image_save(
                    x=X_mnist[num, :],
                    y=reconstucted_target.cpu().numpy(),
                    epoch=epoch + 1,
                    num=num,
                    select=select,
                    rand_select=rand_select,
                    model=model.model_name,
                    lr=lr,
                    tv=TV_strength,
                    scale=scale,
                    # kernel_size=kernel_size,
                    sim=False,
                )
                print(f"mse:{mse_val:.5f}, ssim:{ssim_score:.5f}, PSNR: {psnr:.5f}")
        print(f"\n================ Image {num} の学習終了 ================\n")
        recon_list.append(reconstucted_target.detach().cpu().numpy())
    print("モデルトレーニングが完了しました。")
    return recon_list


def train_gidc(collected_path, target_path, select, rand_select, scale):
    # =============================================
    num_epochs = 2000
    lr = 0.05
    TV_strength = 5e-8
    kernel_size = 3
    # =============================================
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)
    Y_random, Y_mnist = collected_signal(
        path=collected_path, select=select, rand_select=rand_select
    )
    X_random, X_mnist = target_image(
        path=target_path, select=select, rand_select=rand_select
    )
    # Y_mnist = Y_mnist * (-1)
    # Y_random = Y_random * (-1)
    print("======================================")
    # print("Y_mnist shape:", Y_mnist.shape)
    # print("Y_random shape:", Y_random.shape)
    # print("X_mnist shape:", X_mnist.shape)
    # print("X_random shape:", X_random.shape)
    print("X_mnist min, max:", X_mnist.min(), X_mnist.max())
    print("X_random min, max:", X_random.min(), X_random.max())
    print("======================================")
    print("ランダムパターンからspeckle_patternsを推定します。pinvを利用します。")
    S = scale * speckle_pred_inv(
        path_x=target_path,
        path_y=collected_path,
        select=select,
        rand_select=rand_select,
    )
    # S.shape: (28*28, 10000枚)
    print(f"speckle by random scaled by {scale}:", S.min(), S.max(), S.shape)
    print("Y_mnist range:", Y_mnist.min(), Y_mnist.max())
    print("Y_mnist, Y_random:", Y_mnist.shape, Y_random.shape)
    X_mnist_first = img_reconstruction(S, Y_mnist)
    print("X_input:", X_mnist_first.min(), X_mnist_first.max(), X_mnist_first.shape)
    print("======================================")
    S_tensor = np_to_torch(S).float().to(device)
    Y_mnist_tensor = np_to_torch(Y_mnist).float()
    X_input_tensor = np_to_torch(X_mnist_first).float()
    criterion = nn.MSELoss()
    recon_list = []
    for num in range(10):
        print(f"\n================ Image {num} の学習開始 ================\n")
        input = min_max_normalize(X_input_tensor[num].reshape((1, 1, 28, 28))).to(
            device
        )
        # input = standardize(X_input_tensor[num].reshape((1, 1, 28, 28))).to(device)
        y_ = Y_mnist_tensor[num].to(device)
        # print(y_.shape)
        model = GIDC28(kernel_size=kernel_size, name="gidc", select=select).to(device)
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
            if (epoch + 1) % 200 == 0:
                print(
                    f"Image {num}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}"
                )
                model.eval()
                with torch.no_grad():
                    reconstucted_target = (
                        model(input).squeeze(0).squeeze(0).reshape(784)
                    )
                    # print("再構成画像の shape:", reconstucted_target.shape)
                mse_val, ssim_score, psnr = image_save(
                    x=X_mnist[num, :],
                    y=reconstucted_target.cpu().numpy(),
                    epoch=epoch + 1,
                    num=num,
                    select=select,
                    rand_select=rand_select,
                    model=model.model_name,
                    lr=lr,
                    tv=TV_strength,
                    scale=scale,
                    kernel_size=kernel_size,
                    sim=False,
                )
                print(f"mse:{mse_val:.5f}, ssim:{ssim_score:.5f}, PSNR: {psnr:.5f}")
        print(f"\n================ Image {num} の学習終了 ================\n")
        recon_list.append(reconstucted_target.detach().cpu().numpy())
    print("モデルトレーニングが完了しました。")
    return recon_list


# ===================================================================================
# Rough speckle pattern containers
# ===================================================================================


def train_simulation(speckle_path, target_path, select, rand_select, scale):
    # =============================================
    num_epochs = 2000
    lr = 0.1
    TV_strength = 1e-9
    kernel_size = 3
    # =============================================
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)
    # Y_random, Y_mnist = collected_signal(
    #     path=collected_path, select=select, rand_select=rand_select
    # )
    X = load_npz_file(file_name=target_path)
    X_random, X_mnist = target_image(
        path=target_path, select=select, rand_select=rand_select
    )
    print("======================================")
    print("X:", X.shape)
    print("X_mnist shape:", X_mnist.shape)
    print("X_random shape:", X_random.shape)
    print("X_mnist min, max:", X_mnist.min(), X_mnist.max())
    print("X_random min, max:", X_random.min(), X_random.max())
    print("======================================")
    speckle_npz = os.path.join(SPECKLE_DATA_DIR, speckle_path)
    S_ = np.load(speckle_npz)["arr_0"].reshape((784, 784))  # shape: (784枚, 28*28)
    print(S_.shape)
    # X.shape: (6020枚, 28*28)
    Y_all = np.dot(X, S_.T)  # Y_all.shape: (6020, 784)
    print("Y_all shape:", Y_all.shape)
    Y_random, Y_mnist = simulated_signal(
        signal=Y_all, select=select, rand_select=rand_select
    )
    print("Y_mnist, Y_random:", Y_mnist.shape, Y_random.shape)
    S = speckle_pred_simulate(
        path_x=target_path, Y_all=Y_all, select=select, rand_select=rand_select
    )
    print(f"speckle by random scaled by {scale}:", S.min(), S.max(), S.shape)
    print("Y_mnist range:", Y_mnist.min(), Y_mnist.max())
    X_mnist_first = img_reconstruction(S, Y_mnist)
    print("X_first:", X_mnist_first.shape)
    # plt.imshow(X_mnist_first[1].reshape((28, -1)))
    # plt.show()
    print("======================================")
    S_tensor = (np_to_torch(S).float()).to(device)
    print("S_tensor:", S_tensor.min(), S_tensor.max())
    Y_mnist_tensor = np_to_torch(Y_mnist).float()
    X_input_tensor = np_to_torch(X_mnist_first).float()
    criterion = nn.MSELoss()
    recon_list = []

    for num in range(10):
        print(f"\n================ Image {num} の学習開始 ================\n")
        # input = min_max_normalize(X_input_tensor[num].reshape((1, 1, 28, 28))).to(
        #     device
        # )
        input = standardize(X_input_tensor[num].reshape((1, 1, 28, 28))).to(device)
        print("input:", X_input_tensor.min(), X_input_tensor.max())
        print("Y_mnist_tensor:", Y_mnist_tensor.min(), Y_mnist_tensor.max())
        y_ = Y_mnist_tensor[num].to(device)
        # print(y_.shape)
        model = GIDC28(kernel_size=kernel_size, name="gidc", select=select).to(device)
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
            if (epoch + 1) % 200 == 0:
                print(
                    f"Image {num}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}"
                )
                model.eval()
                with torch.no_grad():
                    reconstucted_target = (
                        model(input).squeeze(0).squeeze(0).reshape(784)
                    )
                    # print("再構成画像の shape:", reconstucted_target.shape)
                mse_val, ssim_score, psnr = image_save(
                    x=X_mnist[num, :],
                    y=reconstucted_target.cpu().numpy(),
                    epoch=epoch + 1,
                    num=num,
                    select=select,
                    rand_select=rand_select,
                    model=model.model_name,
                    lr=lr,
                    tv=TV_strength,
                    scale=scale,
                    kernel_size=kernel_size,
                    sim=True,
                )
                print(f"mse:{mse_val:.5f}, ssim:{ssim_score:.5f}, PSNR: {psnr:.5f}")
        print(f"\n================ Image {num} の学習終了 ================\n")
        recon_list.append(reconstucted_target.detach().cpu().numpy())
    print("モデルトレーニングが完了しました。")
    return recon_list


# ===================================================================================
# Supervised trainer functions
# ===================================================================================


# def train_random_supervised(collected_path, target_path, select, rand_select, scale):
#     # =============================================
#     num_epochs = 2000
#     lr = 0.0001
#     TV_strength = 5e-8
#     kernel_size = 0
#     # =============================================
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cpu"
#     print("Using device:", device)
#     Y_random, Y_mnist = collected_signal(
#         path=collected_path, select=select, rand_select=rand_select
#     )
#     X_random, X_mnist = target_image(
#         path=target_path, select=select, rand_select=rand_select
#     )
#     # Y_mnist = Y_mnist * (-1)
#     # Y_random = Y_random * (-1)
#     print("======================================")
#     # print("Y_mnist shape:", Y_mnist.shape)
#     # print("Y_random shape:", Y_random.shape)
#     # print("X_mnist shape:", X_mnist.shape)
#     # print("X_random shape:", X_random.shape)
#     print("X_mnist min, max:", X_mnist.min(), X_mnist.max())
#     print("X_random min, max:", X_random.min(), X_random.max())
#     print("======================================")
#     print("ランダムパターンからspeckle_patternsを推定します。pinvを利用します。")
#     S = scale * speckle_pred_inv(
#         path_x=target_path, path_y=collected_path, select=select
#     )
#     print("speckle by random:", S.min(), S.max(), S.shape)
#     print("======================================")
#     S_tensor = np_to_torch(S).float().to(device)
#     Y_random_tensor = np_to_torch(Y_random).float()
#     X_random_tensor = np_to_torch(X_random).float().to(device)
#     criterion = nn.MSELoss()
#     recon_list = []
#     for num in range(10):
#         print(f"\n================ Random {num} の学習開始 ================\n")
#         y_ = Y_random_tensor[num].to(device)
#         print(y_.shape)
#         model = FCModel(
#             input_size=10000, hidden_size=1024, output_size=784, select=select
#         ).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         for epoch in range(num_epochs):
#             model.train()
#             optimizer.zero_grad()
#             output = model(y_).reshape((1, 1, 28, 28))
#             tv = TV_strength * total_variation_loss(output)
#             Y_dash = torch.mm(output.reshape(1, 28**2), S_tensor)
#             img_loss = criterion(X_random_tensor[num, :], output.reshape(28**2))
#             loss = 0 * criterion(Y_dash, y_.unsqueeze(0)) + img_loss + tv
#             loss.backward()
#             optimizer.step()
#             if (epoch + 1) % 500 == 0:
#                 print(
#                     f"Image {num}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.14f}"
#                 )
#                 model.eval()
#                 with torch.no_grad():
#                     reconstucted_target = model(y_).squeeze(0).squeeze(0).reshape(784)
#                     # print("再構成画像の shape:", reconstucted_target.shape)
#                 mse_val, ssim_score, psnr = image_save(
#                     x=X_random[num, :],
#                     y=reconstucted_target.cpu().numpy(),
#                     epoch=epoch + 1,
#                     num=num,
#                     select=select,
#                     rand_select=rand_select,
#                     model=model.model_name,
#                     lr=lr,
#                     tv=TV_strength,
#                     scale=scale,
#                     kernel_size=0
#                 )
#                 print(f"mse:{mse_val:.5f}, ssim:{ssim_score:.5f}, PSNR: {psnr:.5f}")
#         print(f"\n================ Random {num} の学習終了 ================\n")
#         recon_list.append(reconstucted_target.detach().cpu().numpy())
#     print("モデルトレーニングが完了しました。")
#     return recon_list

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader

# class MyRandomDataset(Dataset):
#     def __init__(self, Y, X):
#         """
#         Y: [N, 10000] の形状 (測定信号)
#         X: [N, 784] の形状 (目標画像: 28x28 → 784)
#         """
#         self.Y = Y
#         self.X = X

#     def __len__(self):
#         return self.Y.shape[0]

#     def __getitem__(self, idx):
#         return self.Y[idx], self.X[idx]

# def train_random_supervised_batch(collected_path, target_path, select, rand_select, scale):
#     """
#     ・ランダムデータ (Y_random, X_random) のみでバッチ学習
#     ・学習完了後、Y_mnist を入力して再構成結果を得る
#     """

#     # =============================================
#     num_epochs = 1000
#     lr = 0.04
#     TV_strength = 0
#     batch_size = 250  # バッチサイズは任意でOK
#     # =============================================

#     # デバイスの設定
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cpu"
#     print("Using device:", device)

#     # ------------------------------------------------------
#     # データの読み込み（Y:信号, X:画像）
#     #   - ランダムデータ (Y_random, X_random)
#     #   - MNIST由来データ (Y_mnist,  X_mnist)
#     # ------------------------------------------------------
#     Y_random, Y_mnist = collected_signal(
#         path=collected_path, select=select, rand_select=rand_select
#     )
#     X_random, X_mnist = target_image(
#         path=target_path, select=select, rand_select=rand_select
#     )

#     print("======================================")
#     print("X_mnist min, max:", X_mnist.min(), X_mnist.max())
#     print("X_random min, max:", X_random.min(), X_random.max())
#     print("======================================")
#     print("ランダムパターンからspeckle_patternsを推定します。pinvを利用します。")

#     # Speckleパターン推定 (scale倍している点に注意)
#     S = scale * speckle_pred_inv(
#         path_x=target_path, path_y=collected_path, select=select
#     )
#     print("speckle by random:", S.min(), S.max(), S.shape)
#     print("======================================")

#     # Tensor化
#     S_tensor = np_to_torch(S).float().to(device)           # shape: [784, 10000]
#     Y_random_tensor = np_to_torch(Y_random).float()        # shape: [N, 10000]
#     X_random_tensor = np_to_torch(X_random).float()        # shape: [N, 784]

#     # MNIST由来の測定信号・画像もTensor化
#     # こちらは学習には使わず、推論のみで使用
#     Y_mnist_tensor = np_to_torch(Y_mnist).float().to(device)   # shape: [N, 10000] (例えば10枚など)
#     X_mnist_tensor = np_to_torch(X_mnist).float().to(device)   # shape: [N, 784]

#     # ------------------------------------------------------
#     # データセット & データローダー (ランダム画像のみ)
#     # ------------------------------------------------------
#     dataset = MyRandomDataset(Y_random_tensor, X_random_tensor)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

#     # ------------------------------------------------------
#     # モデルの定義
#     # ------------------------------------------------------
#     model = FCModel(
#         input_size=10000,
#         hidden_size=1024,
#         output_size=784,
#         select=select
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     # ------------------------------------------------------
#     # 学習ループ (ランダムデータのみ)
#     # ------------------------------------------------------
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = 0.0

#         for y_batch, x_batch in dataloader:
#             y_batch = y_batch.to(device)  # (batch_size, 10000)
#             x_batch = x_batch.to(device)  # (batch_size, 784)

#             optimizer.zero_grad()

#             # 順伝播
#             output = model(y_batch)  # shape: (batch_size, 784)

#             # TV損失
#             output_4d = output.reshape(-1, 1, 28, 28)
#             tv = TV_strength * total_variation_loss(output_4d)

#             # Y_dash = output * S (バッチ分まとめて計算)
#             Y_dash = torch.matmul(output, S_tensor)  # (batch_size, 10000)

#             # 画像再構成の誤差
#             img_loss = 1e-5 * criterion(output, x_batch)

#             # 必要に応じて観測信号Yとの誤差も足す (ここでは0倍)
#             recon_loss = 1 * criterion(Y_dash, y_batch)

#             # 合計損失
#             loss = img_loss + recon_loss + tv

#             # 逆伝播 & 更新
#             loss.backward()
#             optimizer.step()

#             epoch_loss += loss.item() * y_batch.size(0)

#         # ロギング（例: 500エポックごと）
#         if (epoch + 1) % 100 == 0:
#             avg_loss = epoch_loss / len(dataset)
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.14f}")

#             # 必要ならこのタイミングで推論可視化も可能
#             model.eval()
#             with torch.no_grad():
#                 # バッチ最初の1枚などを可視化してもよい
#                 pass

#     # ------------------------------------------------------
#     # 学習終了後、推論を行う
#     # ------------------------------------------------------
#     print("モデルトレーニングが完了しました。")
#     model.eval()

#     # 1) ランダムデータの推論結果 (Y_random → output)
#     recon_list_random = []
#     with torch.no_grad():
#         for i in range(X_random_tensor.shape[0]):
#             y_single = Y_random_tensor[i].unsqueeze(0).to(device)  # shape: [1, 10000]
#             recon = model(y_single).squeeze(0).cpu().numpy()       # shape: [784]
#             recon_list_random.append(recon)

#     # 2) MNIST由来データの推論結果 (Y_mnist → output)
#     recon_list_mnist = []
#     with torch.no_grad():
#         for i in range(Y_mnist_tensor.shape[0]):
#             y_single = Y_mnist_tensor[i].unsqueeze(0)  # shape: [1, 10000]
#             recon = model(y_single).squeeze(0).cpu().numpy()       # shape: [784]
#             recon_list_mnist.append(recon)

#     # 必要に応じて、recon_list_mnist と X_mnist_tensor を比較して指標を計算してもOK
#     # 例: image_save や SSIM/PSNR など

#             image_save(x=X_mnist[i, :],
#                     y=recon,
#                     epoch=epoch + 1,
#                     num=i,
#                     select=select,
#                     rand_select=rand_select,
#                     model=model.model_name,
#                     lr=lr,
#                     tv=TV_strength,
#                     scale=scale,
#                     kernel_size=0)
#     # ランダム画像＆MNIST画像の再構成リストを両方返す例
#     return recon_list_random, recon_list_mnist
