from PIL import Image, ImageFilter, ImageEnhance
import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

filters_list = [('BLUR',ImageFilter.BLUR),
                ('CONTOUR',ImageFilter.CONTOUR),
                ('DETAIL',ImageFilter.DETAIL),
                ('EDGE_ENHANCE',ImageFilter.EDGE_ENHANCE),
                ('EDGE_ENHANCE_MORE',ImageFilter.EDGE_ENHANCE_MORE),
                ('EMBOSS',ImageFilter.EMBOSS),
                ('FIND_EDGES',ImageFilter.FIND_EDGES),
                ('SHARPEN',ImageFilter.SHARPEN),
                ('SMOOTH',ImageFilter.SMOOTH),
                ('SMOOTH_MORE',ImageFilter.SMOOTH_MORE),
                ('brightness', ImageEnhance.Brightness),
                ('contrast', ImageEnhance.Contrast)]

enhance_list = [
                ('brightness', ImageEnhance.Brightness),
                ('contrast', ImageEnhance.Contrast)
                ]


for pil_name, pil_filter in filters_list:
    os.mkdir(f'data/{pil_name}/')
# There are 43 classes of signs. Here I am iterating over each class
# and saving new copies of each picture into the processed images folder.
    src_folder = os.listdir('data/train/')
    for k in src_folder:
        os.mkdir(f'data/{pil_name}/{k}/')
        src_file = os.listdir(f'data/train/{k}')
    
        for img_file in src_file:
            if img_file[-3:] == 'ppm':
                image = Image.open(f'data/train/{k}/{img_file}')
                image.filter(pil_filter).save(f'data/{pil_name}/{k}/{img_file}', 'ppm')

for pil_name, pil_enhancer in enhance_list:
    os.mkdir(f'data/{pil_name}/')
# There are 43 classes of signs. Here I am iterating over each class
# and saving new copies of each picture into the processed images folder.
    src_folder = os.listdir('data/train/')
    for k in src_folder:
        os.mkdir(f'data/{pil_name}/{k}/')
        src_file = os.listdir(f'data/train/{k}')
    
        for img_file in src_file:
            if img_file[-3:] == 'ppm':
                image = Image.open(f'data/train/{k}/{img_file}')
                enhancer = pil_enhancer(image)
                enhancer(.5).save(f'data/{pil_name}/{k}/{img_file}.5', 'ppm')
                enhancer(.25).save(f'data/{pil_name}/{k}/{img_file}.25', 'ppm')

src_folder = os.listdir('data/test/')
for k in src_folder:
    os.mkdir('app/static/test/')

    for img_file in src_file:
        if img_file[-3:] == 'ppm':
            Image.open(f'data/test/{img_file}').save(f'app/static/test{img_file[:-3]}.png')