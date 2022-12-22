import pandas as pd

import PIL
import cv2
import numpy as np
import numpy.random
import skimage
from matplotlib import pyplot as plt
from skimage import io
import imutils
import Constants
import lab2
from lab2 import get_proximity, get_dctn
from PIL import Image
from scipy.ndimage import rotate as rotate_image



def show_tesing_results(x, y, xlabel, ylabel, plt_label):
    plt.plot(x, y, label=plt_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def get_preprocessed_image(C_W, C, param, name_preprocessing):
    if name_preprocessing == 'cut':
        return cut_operation(C_W, C, param)
    if name_preprocessing == 'RotRest':
        return rotate_and_back_operation(C_W, param)
    if name_preprocessing == 'smooth':
        return get_smooth_image(C_W, param)
    if name_preprocessing == 'wh_noise':
        return get_wh_noise_image(C_W, param)


def test_preprocessing(C_W, C, params_array, preprocessing_name, alpha, watermark, watermark_range):
    ro_array = []
    f = get_dctn(C)
    for param in params_array:
        preprocessed_image = get_preprocessed_image(C_W, C, param, preprocessing_name)
        f_W = get_dctn(preprocessed_image)
        ro = get_proximity(f, f_W, alpha, watermark, watermark_range)
        ro_array.append(ro)

    show_tesing_results(params_array, ro_array, 'params', 'ro', 'param/ro')
    plt.title(f'testing {preprocessing_name}')
    plt.show()


def cut_operation(C_W, C, param):
    result = np.copy(C)
    N1 = C_W.shape[0]
    N2 = C_W.shape[1]

    for i in range(0, int(N1 * param)):
        for j in range(0, int(N2 * param)):
            result[i, j] = C_W[i, j]
    return result


def rotate_and_back_operation(C_W, param):
    tmp = rotate_image(C_W, param)
    return rotate_image(tmp, -param)


def get_smooth_image(C_W, param):
    return cv2.blur(C_W, (param, param))


def get_wh_noise_image(C_W, param):
    random_noise = numpy.random.normal(scale=param, size=C_W.shape)
    return C_W + random_noise


names = [
    'cut', 'RotRest', 'smooth', 'wh_noise'
]
params = [
    np.arange(0.2, 0.9, 0.1),
    np.arange(0, 42, 7),
    np.arange(3, 15, 2),
    np.arange(400, 1000, 100)
]


def two_distortion(C, C_W, alpha, watermark, watermark_range, params1, name1, params2, name2):
    ro_array = np.zeros(shape=(len(params1), len(params2)))
    f = get_dctn(C)

    for i in range(0, len(params1)):
        param1 = params1[i]
        tmp_image = get_preprocessed_image(C_W, C, param1, name1)
        for j in range(0, len(params2)):
            param2 = params2[j]
            res_image = get_preprocessed_image(tmp_image, C, param2, name2)
            f_W = get_dctn(res_image)
            ro_array[i, j] = get_proximity(f, f_W, alpha, watermark, watermark_range)

    df = pd.DataFrame(ro_array, columns=params2, index=params1)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

    return ro_array


if __name__ == "__main__":
    start_image = PIL.Image.open("bridge.tif")
    C = io.imread(r"bridge.tif")
    size_watermark = 24576

    watermark_range = Constants.watermark_range

    alpha = Constants.best_alpha
    watermark = np.load('watermark.npy')
    C_W_extracted = io.imread('image_with_watermark.png')

    for name, param in zip(names, params):
        test_preprocessing(C_W_extracted, C, param, name, alpha, watermark, watermark_range)

    ro_array = two_distortion(C, C_W_extracted, alpha, watermark, watermark_range, params[0], names[0], params[1],
                              names[1])
