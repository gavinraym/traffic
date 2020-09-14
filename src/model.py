from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import shutil

class Model():
    '''Create a neural network!'''
    def __init__(self, name):
        self.name = name
        self.model = Sequential()
        self.save = lambda: self.model.save(f'models/{name}')

    def add_layers(self, depth=3, dropout=.25, func='relu'):
        '''Add 2,4,or 6 layers to your network.
        Choose a dropout rate, and an activation function too!'''
        self.model.add(Conv2D(filters=32, kernel_size=(5,5), activation=func, input_shape=(30, 30,3)))

        self.model.add(Conv2D(filters=32, kernel_size=(5,5), activation=func))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(rate=dropout))

        if depth >1:
            self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation=func))
            self.model.add(MaxPool2D(pool_size=(2,2)))
            self.model.add(Dropout(rate=dropout))
        
        if depth == 3:
            self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation=func))
            self.model.add(MaxPool2D(pool_size=(2,2)))
            self.model.add(Dropout(rate=dropout))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation=func))
        self.model.add(Dropout(rate=dropout))
        self.model.add(Dense(43, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.save()

    def fit(self, filters=['train'], epochs=10):
        '''Fit the model with a selection of image preprocessing filter
        and number of epochs'''
        for filter in filters:
            if filter in ('BLUR','CONTOUR','DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE',
                            'EMBOSS','FIND_EDGES','SHARPEN','SMOOTH','SMOOTH_MORE','train'):
                datagen = ImageDataGenerator(rescale=1./255, 
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                rotation_range=45,
                                                horizontal_flip=False,
                                                vertical_flip=False,
                                                validation_split = .25)

                train_generator = datagen.flow_from_directory(
                            f'data/{filter}/', 
                            target_size=(30, 30),
                            batch_size=64,
                            subset='training')

                validation_generator = datagen.flow_from_directory(
                            f'data/{filter}/',
                            target_size=(30, 30),
                            batch_size=64,
                            subset='validation')

                self.model.fit_generator(
                            train_generator,
                            epochs=epochs,
                            validation_data= validation_generator,
                            steps_per_epoch=450,
                            validation_steps=150,
                            shuffle=True,
                            workers=0)
        self.save()
        
    def score(self):
        '''[score_it] function, for evaluating saved models with test images.

        Requires a model name (string) to designate which saved model to use.
        Saved model must reside in models directory in a folder with the 
        same name. Use Keras.models.save_model(model, f'models/{model_name})
        
        Reports are written to /reports/{model_name}/ directory. If previously
        made reports exist, then report directory is numbered like {model_name1}
        
        There is no need to designate test data. This class will use the test
        set located in data/test.'''

        # This block of code predicts on the test images, and compiles the results
        # into a pd.DataFrame called preds. This df is then used to generate the reports 
        preds = pd.DataFrame(columns=['predicted_class', 'actual_class', 'image_location'])
        actual = pd.read_csv('data/test_classes.csv', index_col=0)
        src_file = os.listdir(f'data/test')
        for img_file in src_file:
            # Here, each test image is being predicted on in turn. 
            if img_file[-3:] != 'csv':
                image = Image.open(f'data/test/{img_file}')
                image = np.asarray(image.resize((30,30)))
                image = np.resize(image, [1,30,30,3])
                # Predictions, actual class, and image location are added to preds
                preds = preds.append({
                    'predicted_class':np.argmax(self.model.predict(image)[0], axis=-1),
                    'actual_class':actual.loc[img_file]['ClassId'], 
                    'image_location':img_file
                    }, ignore_index=True)

        # Creates a report csv, which contains the results of the test 
        preds.to_csv(f'reports/{self.name}')


if __name__ == '__main__':
    model = Model('test')
    model.add_layers(depth=2)
    model.fit()
    model.score()




    