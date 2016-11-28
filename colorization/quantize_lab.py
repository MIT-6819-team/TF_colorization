"""Quantize the Lab Color Space"""
from skimage import color
import numpy as np

QUANTIZE_SIZE = 10

def rgb_to_lab(rgb):
  return color.rgb2lab(rgb)

def rgb_color_to_single_pixel_image(rgb):
    return np.array([[list(rgb)]], dtype=np.uint8)

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

print construct_quantization()
