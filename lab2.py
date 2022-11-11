import random

import PIL
import numpy as np
import scipy.fftpack
import skimage.metrics
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io


def generate_watermark(shape: [int, int], key: int, mean=0., spread=1.):
    rng = np.random.default_rng(key)
    return rng.normal(mean, spread, shape)


def get_dctn(array):
    return scipy.fftpack.dctn(array, norm='ortho')


def get_inverse_dctn(feature_array):
    return scipy.fftpack.idctn(feature_array, norm='ortho')


def get_watermark_array(shape, range, watermark):
    array = np.zeros(shape)
    count = 0
    for i in range[0]:
        for j in range[1]:
            array[i, j] += watermark[count]
            count += 1
    return array


def get_extracted_features(f, f_w, alpha, watermark_range):
    features_from_extracted_watermark = []
    for i in watermark_range[0]:
        for j in watermark_range[1]:
            features_from_extracted_watermark.append(
                (f_w[i, j] - f[i, j]) / alpha)

    return np.array(features_from_extracted_watermark)


def get_extracted_features_v2(f, f_w, alpha):
    features_from_extracted_watermark = np.zeros(f.shape)
    for i in range(0, features_from_extracted_watermark.shape[0]):
        for j in range(0, features_from_extracted_watermark.shape[1]):
            features_from_extracted_watermark[i, j] = (f_w[i, j] - f[i, j]) / alpha

    return features_from_extracted_watermark


def insert_watermark(feature_array, alpha, watermark, watermark_range):
    feature_array_with_watermark = np.copy(feature_array)
    watermark_features = np.ravel(get_dctn(watermark))
    count = 0
    for i in watermark_range[0]:
        for j in watermark_range[1]:
            feature_array_with_watermark[i, j] += alpha * watermark_features[count]
            count += 1
    return feature_array_with_watermark


def get_proximity(f, f_w, alpha, watermark_array, watermark_range: [int, int]):
    features_extracted_watermark = get_extracted_features(f, f_w, alpha, watermark_range)
    features_watermark = np.ravel(get_dctn(watermark_array))
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    end_range = len(features_extracted_watermark)
    for n in range(0, end_range):
        sum_1 += features_extracted_watermark[n] * features_watermark[n]
        sum_2 += features_extracted_watermark[n] * features_extracted_watermark[n]
        sum_3 += features_watermark[n] * features_watermark[n]
    return sum_1 / (np.sqrt(sum_2) * np.sqrt(sum_3))


def get_proximity_v2(f, f_w, alpha, watermark_array, watermark_range):
    features_extracted_watermark = get_extracted_features_v2(f, f_w, alpha)
    features_watermark = np.ravel(get_dctn(watermark_array))
    count = 0
    sum_1 = 0
    sum_2 = 0
    sum_3 = 0
    for i in watermark_range[0]:
        for j in watermark_range[1]:
            sum_1 += features_extracted_watermark[i, j] * features_watermark[count]
            sum_2 += features_extracted_watermark[i, j] * features_extracted_watermark[i, j]
            sum_3 += features_watermark[count] * features_watermark[count]
            count += 1
    return sum_1 / (np.sqrt(sum_2) * np.sqrt(sum_3))


def get_optimal_alpha(alpha_first, increment, image, watermark, watermark_range, ro_threeshold, psnr_threeshold):
    alpha = alpha_first
    iteration_count = 0
    f = get_dctn(image)
    while True:
        print(f"alpha = {alpha}")
        f_W = insert_watermark(feature_array=f, alpha=alpha,
                               watermark=watermark, watermark_range=watermark_range)
        C_W_extracted = get_inverse_dctn(f_W)
        io.imsave('image_with_watermark.png', C_W_extracted)
        io.show()

        image_with_watermark = io.imread('image_with_watermark.png')
        f_W_extracted = get_dctn(image_with_watermark)
        ro = get_proximity(f, f_W_extracted, alpha, watermark, watermark_range)
        psnr = skimage.metrics.peak_signal_noise_ratio(image, image_with_watermark)
        if ro >= ro_threeshold and psnr >= psnr_threeshold:
            return alpha, psnr
        iteration_count += 1
        print(f"iteration number {iteration_count} alpha = {alpha} ro = {ro} psnr = {psnr}")
        alpha += increment


if __name__ == '__main__':
    """
    Реализовать генерацию ЦВЗ 𝛺 как псевдослучайной последовательности
    заданной длины из чисел, распределённых по нормальному закону
    """

    start_image = PIL.Image.open("bridge.tif")
    C = io.imread(r"bridge.tif").astype(int)
    size_watermark = 24576

    i_range = range(0, 192)
    j_range = range(128, 256)
    watermark_range = [i_range, j_range]
    watermark_shape = [192, 128]
    watermark = generate_watermark(shape=watermark_shape, key=1)

    """
    Реализовать трансформацию исходного контейнера к пространству признаков согласно варианту задания.
    """
    f = get_dctn(C)
    extracted_image = get_inverse_dctn(f)
    """
    Осуществить встраивание информации методом, определяемым
    вариантом задания. Значения параметра встраивания устанавливается произвольным образом
    """

    alpha = random.random()
    print(f"alpha = {alpha}")
    f_W = insert_watermark(feature_array=f, alpha=alpha,
                           watermark=watermark, watermark_range=watermark_range)
    C_W_extracted = get_inverse_dctn(f_W)
    io.imshow(C_W_extracted, cmap='gray')
    io.imsave('image_with_watermark.png', C_W_extracted)
    io.show()
    """
    Считать носитель информации из файла и повторно выполнить п. 2 для носителя информации
    """
    image_with_watermark = io.imread('image_with_watermark.png')
    f_W_extracted = get_dctn(image_with_watermark)
    ro = get_proximity(f, f_W_extracted, alpha, watermark, watermark_range)
    print(f"ro = {ro}")
    optimal_alpha, psnr = get_optimal_alpha(0.2, 0.1, C, watermark, watermark_range, 0.9, 30.0)
    print(f"optimal_alpha = {optimal_alpha} psnr = {psnr}")

    """
    Ложное обнаружение
    """
    ro_array = []
    ro_array.append(ro)
    for i in range(0, 100):
        new_watermark = generate_watermark(watermark_shape, i)
        ro_array.append(get_proximity(f, f_W_extracted, alpha, new_watermark, watermark_range))

    print(ro_array)
    x = np.arange(0, 101)
    fig, ax = plt.subplots()
    ax.plot(x, np.array(ro_array))
    plt.show()