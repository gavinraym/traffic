from PIL import Image, ImageFilter
import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


orig = list()
contour = list()
edge = list()
edge2 = list()
emboss = list()
gaussian = list()
y = list()

# There are 43 classes of signs. Here I am iterating over each class
# and saving new copies of each picture into the processed images folder.
for k in range(43):
    src_file = os.listdir(f'data/original_data/{k}')
    
    for img_file in src_file:
        if img_file[-3:] != 'csv':
            image = Image.open(f'data/original_data/{k}/{img_file}')

            # Resizing images to all be 30 x 30 
            # (This is the recommended size for this set)
            image = image.resize((30,30))
            orig.append(np.array(image))
            contour.append(np.array(image.filter(ImageFilter.CONTOUR)))
            edge.append(np.array(image.filter(ImageFilter.EDGE_ENHANCE_MORE)))
            edge2.append(np.array(image.filter(ImageFilter.FIND_EDGES)))
            emboss.append(np.array(image.filter(ImageFilter.EMBOSS)))
            gaussian.append(np.array(image.filter(ImageFilter.GaussianBlur)))
            y.append(k)


np.save(f'data/training_data/orig', np.asarray(orig))
np.save(f'data/training_data/contour', np.asarray(contour))
np.save(f'data/training_data/edge', np.asarray(edge2))
np.save(f'data/training_data/edge2', np.asarray(edge2))
np.save(f'data/training_data/emboss', np.asarray(emboss))
np.save(f'data/training_data/gaussian', np.asarray(gaussian))

y = np.asarray(y).reshape(-1,1)
np.save('data/training_data/y_train', to_categorical(y))

