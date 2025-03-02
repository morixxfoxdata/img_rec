import numpy as np
import matplotlib.pyplot as plt
import torch
from src.dataset.process import load_npz_file, load_npz_signal, target_image, collected_signal
from src.dataset.speckle_pred import speckle_pred_inv
from src.utils.inv_recon import img_reconstruction
from src.trainer import train_simple, train_gidc
file_y = "Rand+Mnist+Rand_pix28x28_image(1500+10+1500)x2_sig2500x4wave.npz"
file_x = "Rand+Mnist+Rand_size28x28_image(1500+10+1500)x2.npz"


def main():
    # reconstructed = train_simple(collected_path=file_y, target_path=file_x, select="black", rand_select="both", scale=1)
    reconstructed = train_gidc(collected_path=file_y, target_path=file_x, select="black", rand_select="both", scale=1)

if __name__ == "__main__":
    main()