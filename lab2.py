import math

import PIL
import numpy as np
import scipy.fftpack
import skimage
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io
import skimage.metrics

def generate_watermark(size: int, key: int, mean=0., spread=1.):
    rng = np.random.default_rng(key)
    return rng.normal(mean, spread, size)


def get_dct(array):
    return scipy.fftpack.dct(array)


def get_inverse_dct(feature_array):
    return scipy.fftpack.idct(feature_array)


def get_watermark_array(shape, i_range, j_range, watermark):
    array = np.zeros(shape)
    count = 0
    for i in i_range:
        for j in j_range:
            array[i, j] += watermark[count]
            count += 1
    return array


# def insert_watermark(feature_array, alpha, watermark_features_array):
#     feature_array += alpha * watermark_features_array
#     return feature_array

def get_extracted_features(extracted_image, original_image, alpha, i_range, j_range):
    features_from_original_image = get_dct(original_image)
    features_from_extracted_image = get_dct(extracted_image)
    features_from_extracted_watermark = []
    for i in i_range:
        for j in j_range:
            features_from_extracted_watermark.append(
                (features_from_extracted_image[i, j] - features_from_original_image[i, j]) / alpha)

    return np.array(features_from_extracted_watermark)


def insert_watermark(feature_array, alpha, watermark_features_array, i_range, j_range):
    feature_array_with_watermark = np.copy(feature_array)
    count = 0
    for i in i_range:
        for j in j_range:
            feature_array_with_watermark[i, j] += alpha * watermark_features_array[count]
            count += 1
    return feature_array_with_watermark


def get_proximity(image_array, image_with_watermark_array, alpha, watermark_array, i_range, j_range):
    features_extracted_watermark = get_extracted_features(image_with_watermark_array, image_array, alpha, i_range,
                                                          j_range)
    features_watermark = get_dct(watermark_array)
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for n in range(0, features_extracted_watermark.size):
        sum_1 += features_extracted_watermark[n] * features_watermark[n]
        sum_2 += features_extracted_watermark[n] * features_extracted_watermark[n]
        sum_3 += features_watermark[n] * features_watermark[n]
    return sum_1 / (np.sqrt(sum_2) * np.sqrt(sum_3))


if __name__ == '__main__':
    # Реализовать генерацию ЦВЗ 𝛺 как псевдослучайной последовательности
    # заданной длины из чисел, распределённых по нормальному закону
    start_image = PIL.Image.open("bridge.tif")
    image = io.imread(r"bridge.tif").astype(int)
    # c = 0.5
    # size = image.shape[0] * image.shape[1] * c
    size_watermark = 24576
    watermark = generate_watermark(size=size_watermark, key=1)

    # Реализовать трансформацию исходного контейнера к пространству признаков согласно варианту задания.
    features_image = get_dct(image)

    # image_from_idct = get_inverse_dct(features)
    # io.imshow(image_from_idct)
    # io.show()

    # Осуществить встраивание информации методом, определяемым
    # вариантом задания. Значения параметра встраивания устанавливается произвольным образом
    i_range = range(0, 192)
    j_range = range(128, 256)
    alpha = 7.2
    features_watermark_array = get_dct(watermark)
    features_with_watermark = insert_watermark(feature_array=features_image, alpha=alpha,
                                               watermark_features_array=features_watermark_array, i_range=i_range,
                                               j_range=j_range)
    new_image = get_inverse_dct(features_with_watermark)
    io.imshow(new_image, cmap='gray')
    io.imsave('image_with_watermark.tif', new_image)

    # Считать носитель информации из файла и повторно выполнить п. 2 для носителя информации
    image_with_watermark = io.imread('image_with_watermark.tif').astype(int)

    ro = get_proximity(image, image_with_watermark, alpha, watermark, i_range=i_range, j_range=j_range)
    print(ro)

    ro_array = []
    ro_array.append(ro)

    io.show()
    alpha = 0.5
    features_image = get_dct(image)
    features_watermark_array = get_dct(watermark)
    best_alpha = 0
    best_ro = 0
    best_psnr = 0
    for i in range(0, 100):
        features_with_watermark = insert_watermark(feature_array=features_image, alpha=alpha,
                                                   watermark_features_array=features_watermark_array, i_range=i_range,
                                                   j_range=j_range)
        new_image = get_inverse_dct(features_with_watermark)
        ro = get_proximity(image, new_image, alpha, watermark, i_range=i_range, j_range=j_range)
        psnr = skimage.metrics.peak_signal_noise_ratio(image, new_image)
        print(f"iteration № {i}  alpha = {alpha}  ro = {ro}  psnr = {psnr}")
        if ro >= 0.9:
            best_alpha = alpha
            best_ro = ro
            best_psnr = psnr
            break
        alpha += 0.1
    print(f"best alpha = {best_alpha}  best ro = {ro}  best psnr = {best_psnr}")

    # ложное обнаружение
    for i in range(0, 100):
        test_watermark = generate_watermark(size_watermark, key=1)
        features_watermark_array = get_dct(test_watermark)
        features_with_watermark = insert_watermark(feature_array=features_image, alpha=best_alpha,
                                                   watermark_features_array=features_watermark_array, i_range=i_range,
                                                   j_range=j_range)
        new_image = get_inverse_dct(features_with_watermark)
        ro = get_proximity(image, new_image, alpha, watermark, i_range=i_range, j_range=j_range)
        ro_array.append(ro)
    print(ro_array)
    x = np.arange(0, 101)
    fig, ax = plt.subplots()
    ax.plot(x, np.array(ro_array))
    plt.show()
