import numpy as np

def img_reconstruction(S, Y):
    """
    S: size(784, 2500)
    Y: size(10, 2500)
    """
    S_pinv = np.linalg.pinv(S)
    rec_img = np.dot(Y, S_pinv)
    return rec_img