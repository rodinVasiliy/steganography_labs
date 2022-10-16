import copy

import numpy as np
import PIL
import scipy.fftpack
from PIL import Image, ImageDraw
import scipy as sp
from scipy import fftpack, stats, signal
import matplotlib.pyplot as plt
from skimage import io
from random import random


def generate_watermark(size: int, key: int, mean=0., spread=1.):
    rng = np.random.default_rng(key)
    return rng.normal(mean, spread, size)


def get_dct(array):
    return scipy.fftpack.dct(array)


def get_inverse_dct(feauture_array):
    return scipy.fftpack.idct(feauture_array)


def get_watermark_array(shape, i_range, j_range, watermark):
    array = np.zeros(shape)
    count = 0
    for i in i_range:
        for j in j_range:
            array[i, j] += watermark[count]
            count += 1
    return array


def insert_watermark(feature_array, alpha, watermark_array):
    feature_array += alpha * watermark_array
    return feature_array


def get_proximity(image_array, image_with_watermark_array, alpha, watermark_array):
    features_from_image = get_dct(image_array)
    features_from_image_with_watermark = get_dct(image_with_watermark_array)
    feautures_extracted_watermark = (features_from_image_with_watermark - features_from_image) / (alpha * features_from_image)
    feautures_extracted_watermark_ravel = np.ravel(feautures_extracted_watermark)
    feautures_watermark = get_dct(watermark_array)
    feautures_watermark_ravel = np.ravel(feautures_watermark)
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in range(0, feautures_extracted_watermark_ravel.size):
        sum_1 += feautures_extracted_watermark_ravel[i] * feautures_watermark_ravel[i]
        sum_2 += feautures_extracted_watermark_ravel[i] * feautures_extracted_watermark_ravel[i]
        sum_3 += feautures_watermark_ravel[i] * feautures_watermark_ravel[i]
    return sum_1 / (np.sqrt(sum_2) * np.sqrt(sum_3))


if __name__ == '__main__':
    size_watermark = 24576
    watermark = generate_watermark(size=size_watermark, key=1)

    start_image = PIL.Image.open("bridge.tif")
    image = io.imread(r"bridge.tif").astype(int)
    features = get_dct(image)

    # image_from_idct = get_inverse_dct(features)
    # io.imshow(image_from_idct)
    # io.show()

    i_range = range(0, 192)
    j_range = range(128, 256)
    watermark_array = get_watermark_array(image.shape, i_range, j_range, watermark)
    alpha = 0.9
    features_watermark_array = get_dct(watermark_array)
    features_with_watermark = insert_watermark(feature_array=features, alpha=alpha, watermark_array=features_watermark_array)
    new_image = get_inverse_dct(features_with_watermark)
    io.imshow(new_image, cmap='gray')
    io.imsave('image_with_watermark.tif', new_image)

    image_with_watermark = io.imread('image_with_watermark.tif').astype(int)
    features_with_watermark = get_dct(image_with_watermark)

    ro = get_proximity(image, image_with_watermark, alpha, watermark_array)
    print(ro)

    io.show()
