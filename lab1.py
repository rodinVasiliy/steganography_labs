import cv2
import numpy as np


def get_channel(image, channel):
    b, g, r = cv2.split(image)
    if channel == 'blue':
        return b
    if channel == 'green':
        return g
    if channel == 'red':
        return r


def get_plane(channel_image, plane_num):
    return channel_image & (2 ** (plane_num - 1))


def encode_svi1(image, watermark, channel_color, bit_num):
    num_for_clear_bit_plate = 255 - (2 ** (bit_num - 1))

    prepared_watermark = ((watermark / 255) * (2 ** (bit_num - 1))).astype(np.uint8)
    watermark_channel = get_channel(prepared_watermark, channel_color)

    image_with_empty_bit = get_channel(image, channel_color) & num_for_clear_bit_plate

    result_image = image_with_empty_bit | watermark_channel

    r = get_channel(baboon_image, 'red')
    g = get_channel(baboon_image, 'green')
    b = get_channel(baboon_image, 'blue')

    if channel_color == 'blue':
        return cv2.merge([result_image, g, r])
    if channel_color == 'red':
        return cv2.merge([b, g, result_image])
    if channel_color == 'green':
        return cv2.merge([b, result_image, r])


def decode_svi1(encoded_image, channel_color, bit_num):
    encoded_image_channel = get_channel(encoded_image, channel_color)
    return get_plane(encoded_image_channel, bit_num)


def encode_svi4(image, watermark, channel_color, delta):
    h, w, channels = image.shape
    noise = np.empty((h, w), dtype="uint8")
    cv2.randn(noise, 0, delta - 1)

    cv2.imshow("Noise", noise)

    extracted_channel = get_channel(image, channel_color)
    binary_watermark = get_channel(watermark, channel_color)
    changed_channel = (extracted_channel // (2 * delta) * (2 * delta)) + binary_watermark * delta + noise

    r = get_channel(image, 'red')
    g = get_channel(image, 'green')
    b = get_channel(image, 'blue')

    if channel_color == 'blue':
        return noise, cv2.merge([changed_channel, g, r])
    if channel_color == 'red':
        return noise, cv2.merge([b, g, changed_channel])
    if channel_color == 'green':
        return noise, cv2.merge([b, changed_channel, r])


def decode_svi4(encoded_image, original_image, noise, channel_color, delta):
    encoded_image_channel = get_channel(encoded_image, channel_color)
    original_image_channel = get_channel(original_image, channel_color)
    return (encoded_image_channel - noise - (original_image_channel // (2 * delta) * 2 * delta)) / delta


if __name__ == '__main__':
    baboon_image = cv2.imread('baboon.tif')
    ornament = cv2.imread('ornament.tif')
    cv2.imshow("Original", baboon_image)
    svi_1_result = encode_svi1(baboon_image, ornament, 'green', 3)
    svi_1_decode = decode_svi1(svi_1_result, 'green', 3)

    cv2.imshow("Original", baboon_image)
    cv2.imshow("SVI-1 Encoded", svi_1_result)
    cv2.imshow("SVI-1 Decoded", svi_1_decode)

    cv2.waitKey(0)
    VAR = 5
    DELTA = 4 + (4 * VAR) % 3

    result_noise, svi_4_result = encode_svi4(baboon_image, ornament, 'red', DELTA)
    svi_4_decode = decode_svi4(svi_4_result, baboon_image, result_noise, 'red', DELTA)

    cv2.imshow("Original", baboon_image)
    cv2.imshow("SVI-4 Encoded", svi_4_result)
    cv2.imshow("SVI-4 Decoded", svi_4_decode)

    cv2.waitKey(0)
