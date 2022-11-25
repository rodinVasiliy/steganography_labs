import PIL
import numpy as np
from scipy import fft
import skimage.metrics
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io


def prepare_watermark(watermark, i_range, j_range, array_shape):
    result_array = np.zeros(array_shape)
    l = 0
    for i in i_range:
        k = 0
        for j in j_range:
            result_array[i, j] = watermark[l, k]
            k += 1
        l += 1
    return result_array


def generate_watermark(shape: (int, int), key: int, mean=0., spread=1.):
    rng = np.random.default_rng(key)
    return rng.normal(mean, spread, shape)


def get_dct2(array):
    return fft.dctn(array)


def get_inverse_dct2(feature_array):
    return fft.idctn(feature_array)


def get_watermark_array(shape, key: int, mean=0., spread=1.):
    rng = np.random.default_rng(key)
    return rng.normal(mean, spread, shape)


def get_extracted_features(extracted_image, original_image, alpha, i_range, j_range, watermark_shape):
    features_from_original_image = get_dct2(original_image)
    features_from_extracted_image = get_dct2(extracted_image)
    features_from_extracted_watermark = np.zeros(watermark_shape)
    i_w = 0
    j_w = 0
    for i in i_range:
        j_w = 0
        for j in j_range:
            features_from_extracted_watermark[i_w, j_w] = (features_from_extracted_image[i, j] -
                                                           features_from_original_image[i, j]) / alpha
            j_w += 1
        i_w += 1

    return features_from_extracted_watermark


def insert_watermark(feature_array, alpha, watermark_features_array, i_range, j_range):
    result = np.copy(feature_array)
    prepared_watermark = prepare_watermark(watermark_features_array, i_range, j_range,
                                           result.shape)
    for i in i_range:
        for j in j_range:
            result[i, j] += alpha * prepared_watermark[i, j]
    return result


def get_proximity(image_array, image_with_watermark_array, alpha, watermark_array, i_range, j_range):
    features_extracted_watermark = get_extracted_features(image_with_watermark_array, image_array, alpha, i_range,
                                                          j_range, watermark_array.shape)
    features_watermark = get_dct2(watermark_array)

    features_extracted_watermark = np.ravel(features_extracted_watermark)
    features_watermark = np.ravel(features_watermark)

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

    watermark_shape = (192, 128)
    watermark = generate_watermark(watermark_shape, key=1)

    # Реализовать трансформацию исходного контейнера к пространству признаков согласно варианту задания.
    features_image = get_dct2(image)

    # Осуществить встраивание информации методом, определяемым
    # вариантом задания. Значения параметра встраивания устанавливается произвольным образом
    i_range = range(0, 192)
    j_range = range(128, 256)
    alpha = 0.1
    features_watermark = get_dct2(watermark)

    features_with_watermark = insert_watermark(feature_array=features_image, alpha=alpha,
                                               watermark_features_array=features_watermark, i_range=i_range,
                                               j_range=j_range)
    new_image = get_inverse_dct2(features_with_watermark).astype(int)
    io.imshow(new_image, cmap='gray')
    io.imsave('image_with_watermark.png', new_image)

    # Считать носитель информации из файла и повторно выполнить п. 2 для носителя информации
    image_with_watermark = io.imread('image_with_watermark.png').astype(int)

    ro = get_proximity(image, image_with_watermark, alpha, watermark, i_range=i_range, j_range=j_range)
    print(ro)

    ro_array = []
    ro_array.append(ro)

    io.show()
    alpha = 0.01
    features_image = get_dct2(image)
    features_watermark = get_dct2(watermark)
    best_alpha = 0.0
    best_ro = 0.0
    best_psnr = 0.0
    for i in range(0, 100):
        features_with_watermark = insert_watermark(feature_array=features_image, alpha=alpha,
                                                   watermark_features_array=features_watermark, i_range=i_range,
                                                   j_range=j_range)
        new_image = get_inverse_dct2(features_with_watermark).astype(int)
        ro = get_proximity(image, new_image, alpha, watermark, i_range=i_range, j_range=j_range)
        psnr = skimage.metrics.peak_signal_noise_ratio(image, new_image)
        print(f"iteration № {i}  alpha = {alpha}  ro = {ro}  psnr = {psnr}")
        if ro >= 0.9:
            best_alpha = alpha
            best_ro = ro
            best_psnr = psnr
            break
        alpha += 0.01
    print(f"best alpha = {best_alpha}  best ro = {ro}  best psnr = {best_psnr}")
    alpha = 0.1
    # ложное обнаружение
    for i in range(0, 100):
        test_watermark = generate_watermark(watermark_shape, key=i + 5)
        ro = get_proximity(image, new_image, alpha, test_watermark, i_range=i_range, j_range=j_range)
        ro_array.append(ro)
    print(ro_array)
    x = np.arange(0, 101)
    fig, ax = plt.subplots()
    ax.plot(x, np.array(ro_array))
    plt.show()
