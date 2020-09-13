from keras import models
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import sys


def score_it(model_name):

    model = models.load_model(f'models/{model_name}')
    report = open(f'reports/{model_name}.txt', 'w+')
    report.write(f'***Model Report - {model_name} ***\n{"*"*25}\n\n')

    predictions = pd.DataFrame(columns=['predicted_class', 'actual_class', 'image_location'])
    actual = pd.read_csv('data/test_classes.csv', index_col=0)
    src_file = os.listdir(f'data/test')

    for img_file in src_file:
        if img_file[-3:] != 'csv':
            
            image = Image.open(f'data/test/{img_file}')
            image = np.asarray(image.resize((30,30)))
            image = np.resize(image, [1,30,30,3])
            predictions = predictions.append({
                'predicted_class':np.argmax(model.predict(image)[0], axis=-1),
                'actual_class':actual.loc[img_file]['ClassId'], 
                'image_location':img_file
                }, ignore_index=True)
            
            
    falsities = predictions[predictions['predicted_class']!=predictions['actual_class']]
    
    report.write(f'Accuracy = {len(falsities)/len(predictions)}')

    #conf = confusion_matrix(
    #    y_true = predictions['actual_class'],
    #    y_pred = predictions['predicted_class'],
    #    normalize = 'true'
    #    )
    return predictions, actual, falsities#, conf

if __name__ == "__main__":
    result = score_it('train')
