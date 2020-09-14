from PIL import Image, ImageFilter
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
                ('SMOOTH_MORE',ImageFilter.SMOOTH_MORE)]

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
