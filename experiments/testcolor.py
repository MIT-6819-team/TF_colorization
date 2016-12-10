from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color

lab = LabColor(0.903, 16.296, -2.22)
rgb = convert_color(lab, sRGBColor)
print rgb
