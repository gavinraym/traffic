from PIL import Image, ImageFilter
import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

filters_list = [('blur',ImageFilter.BLUR),
                ('contour',ImageFilter.CONTOUR),
                ('detail',ImageFilter.DETAIL),
                ('edge',ImageFilter.EDGE_ENHANCE_MORE),
                ('emboss',ImageFilter.EMBOSS),
                ('edges',ImageFilter.FIND_EDGES),
                ('sharpen',ImageFilter.SHARPEN),
                ('smooth',ImageFilter.SMOOTH_MORE)]

for pil_name, pil_filter in filters_list:
    os.mkdir(f'data/{pil_name}/')
# There are 43 classes of signs. Here I am iterating over each class
# and saving new copies of each picture into the processed images folder.
    for k in range(43):
        os.mkdir(f'data/{pil_name}/{k}/')
        src_file = os.listdir(f'data/train/{k}')
    
        for img_file in src_file:
            if img_file[-3:] != 'csv':
                image = Image.open(f'data/train/{k}/{img_file}')
                image.filter(pil_filter).save(f'data/{pil_name}/{k}/{img_file}', 'ppm')
