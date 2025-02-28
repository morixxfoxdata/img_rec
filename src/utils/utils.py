import torch

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


def total_variation_loss(x):
    # 縦方向の差分 |x[i, j+1] - x[i, j]|
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

    # 横方向の差分 |x[i+1, j] - x[i, j]|
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

    return torch.sum(tv_h) + torch.sum(tv_w)