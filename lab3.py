import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage import io
import imutils
import Constants
from lab2 import get_proximity, get_dctn
from PIL import Image
from scipy.ndimage import rotate as rotate_image


def show(x, y, xlabel, ylabel, plt_label):
    plt.plot(x, y, label=plt_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def cut_operation(C_W, C, param):
    result = np.copy(C)
    N1 = C_W.shape[0]
    N2 = C_W.shape[1]

    for i in range(0, int(N1 * param)):
        for j in range(0, int(N2 * param)):
            result[i, j] = C_W[i, j]
    return result


def test_cut_operation(image_with_watermark, image, watermark, watermark_range, alpha):
    params_array = np.arange(0.2, 0.9, 0.1)
    ro_array = []
    psnr_array = []
    f = get_dctn(image)
    for param in params_array:
        cut_image = cut_operation(image_with_watermark, image, param)
        f_W = get_dctn(cut_image)
        ro = get_proximity(f, f_W, alpha, watermark, watermark_range)
        ro_array.append(ro)
        psnr_array.append(skimage.metrics.peak_signal_noise_ratio(cut_image, image_with_watermark))

    show(params_array, ro_array, 'params', 'ro', 'param/ro')
    plt.show()

    show(params_array, psnr_array, 'params', 'psnr', 'param/psnr')
    plt.show()


def rotate_and_back_operation(C_W, param):
    tmp = rotate_image(C_W, param)
    return rotate_image(tmp, -param)


def test_rot_rest(image_with_watermark, image, watermark, watermark_range, alpha):
    params_array = np.arange(0, 42, 7)
    ro_array = []
    f = get_dctn(image)
    for param in params_array:
        rot_rest_image = rotate_and_back_operation(image_with_watermark, param)
        f_W = get_dctn(rot_rest_image)
        ro = get_proximity(f, f_W, alpha, watermark, watermark_range)
        ro_array.append(ro)

    show(params_array, ro_array, 'params', 'ro', 'param/ro')
    plt.show()


def get_smooth_image(image, param):
    return cv2.blur(image, (param, param))


def test_rot_rest(image_with_watermark, image, watermark, watermark_range, alpha):
    params_array = np.arange(0, 42, 7)
    ro_array = []
    f = get_dctn(image)
    for param in params_array:
        rot_rest_image = rotate_and_back_operation(image_with_watermark, param)
        f_W = get_dctn(rot_rest_image)
        ro = get_proximity(f, f_W, alpha, watermark, watermark_range)
        ro_array.append(ro)

    show(params_array, ro_array, 'params', 'ro', 'param/ro')
    plt.show()



if __name__ == "__main__":
    image_with_watermark = io.imread('image_with_watermark.png')
    image = io.imread(r"bridge.tif").astype(int)
    watermark = np.load('watermark.npy')
    watermark_range = Constants.watermark_range
    alpha = Constants.best_alpha

    # cut
    test_cut_operation(image_with_watermark, image, watermark, watermark_range, alpha)

    # RotRest
    test_rot_rest(image_with_watermark, image, watermark, watermark_range, alpha)
