"""Quantize the Lab Color Space"""
from skimage import color
import numpy as np
import cv2

QUANTIZE_SIZE = 10

def lab_to_rgb(lab):
    return color.lab2rgb(lab)

def rgb_to_lab(rgb):
    return color.rgb2lab(rgb)

def rgb_color_to_single_pixel_image(rgb):
    return np.array([[list(rgb)]], dtype=np.uint8)

def lab_color_to_single_pixel_image(lab):
    return np.array([[list(lab)]], dtype=np.float64)

def construct_quantization():
    colorspace = {}
    for r in range(0,256):
        print r, len(colorspace)
        for g in range(0,256):
            for b in range(0,256):
                rgb = [r, g, b]
                pixel = rgb_color_to_single_pixel_image(rgb)
                lab = rgb_to_lab(pixel)[0][0]
                if int(lab[0]) == 50:
                    print len(colorspace)
                    quantized_lab = (int(lab[0]/QUANTIZE_SIZE), int(lab[1]/QUANTIZE_SIZE), int(lab[2]/QUANTIZE_SIZE))
                    if quantized_lab not in colorspace:
                        colorspace[quantized_lab] = rgb
    print colorspace

def construct_quantization_from_lab():
    colorspace = {}
    SIZE = 140
    viz = np.zeros((SIZE*2, SIZE*2, 3))
    for l in range(100, 0, -1):
        print l
        for a in range(-SIZE, SIZE, QUANTIZE_SIZE):
            for b in range(-SIZE, SIZE, QUANTIZE_SIZE):
                lab = [l, a, b]
                pixel = np.array(lab_color_to_single_pixel_image(lab))
                rgb = lab_to_rgb(pixel)
                recover_lab = rgb_to_lab(rgb)[0][0]

                if approx_equal(recover_lab[0], l) and approx_equal(recover_lab[1], a) and approx_equal(recover_lab[2], b):
                    rgb = rgb[0][0]
                    bgr = (rgb[2], rgb[1], rgb[0])
                    # viz[SIZE + int(a/QUANTIZE_SIZE)*QUANTIZE_SIZE][SIZE + int(b/QUANTIZE_SIZE)*QUANTIZE_SIZE] = bgr
                    # This Lab value is in the RGB colorspace
                    quantized_lab = (int(a/QUANTIZE_SIZE), int(b/QUANTIZE_SIZE))
                    if quantized_lab not in colorspace:
                        cv2.rectangle(viz, (SIZE + int(b/QUANTIZE_SIZE)*QUANTIZE_SIZE, SIZE + int(a/QUANTIZE_SIZE)*QUANTIZE_SIZE),
                        (SIZE + (int(b/QUANTIZE_SIZE) +1)*QUANTIZE_SIZE, SIZE + (int(a/QUANTIZE_SIZE)+1)*QUANTIZE_SIZE),
                        bgr, -1)
                        colorspace[quantized_lab] = rgb
                        print quantized_lab, len(colorspace)

    cv2.imwrite('abspace.jpg', viz*255)
    cv2.imshow('space', viz)
    cv2.waitKey(0)
    return colorspace

def construct_quantization_layer():


def approx_equal(x, y, threshold = 0.5):
    return abs(x - y) < threshold

print construct_quantization_from_lab()

# a = 999999
# b = 999999

# l = 50
# lab = [l, a, b]
# pixel = np.array(lab_color_to_single_pixel_image(lab))
# rgb = lab_to_rgb(pixel)
# recover_lab = rgb_to_lab(rgb)
# print rgb, lab, recover_lab
