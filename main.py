from src.trainer import train_simple

file_y = "Rand+Mnist+Rand_pix28x28_image(1500+10+1500)x2_sig2500x4wave.npz"
file_x = "Rand+Mnist+Rand_size28x28_image(1500+10+1500)x2.npz"
file_s = "mask_patterns_length10_core1e-05.npz"


def main():
    reconstructed = train_simple(
        collected_path=file_y,
        target_path=file_x,
        select="black",
        rand_select="both",
        scale=1,
    )
    # reconstructed = train_gidc(
    #     collected_path=file_y,
    #     target_path=file_x,
    #     select="white",
    #     rand_select="white",
    #     scale=1,
    # )
    # train_simulation(
    #     speckle_path=file_s,
    #     target_path=file_x,
    #     select="black",
    #     rand_select="black",
    #     scale=1,
    # )
    # reconstructed = train_random_supervised(
    #     collected_path=file_y,
    #     target_path=file_x,
    #     select="both",
    #     rand_select="both",
    #     scale=1,
    # )
    # recon_list_random, recon_list_mnist = train_random_supervised_batch(
    #     collected_path=file_y,
    #     target_path=file_x,
    #     select="both",
    #     rand_select="both",
    #     scale=1,
    # )


if __name__ == "__main__":
    main()
