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
from datetime import datetime
import time
import json
import pickle
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns

class Model():
    '''Create a neural network!'''
    def __init__(self, name, description='', model=Sequential()):
        self.index = len(os.listdir('models/'))
        self.name = name
        self.timestamp = datetime.now().strftime('%m/%d %H:%M')
        self.model_path = f'models/{self.index}_{self.name}'
        self.pickle_path = f'pickles/{self.index}_{self.name}.pk'
        self.description = description
        self.filters = list()
        self.graph = None
        self.preds = pd.read_csv('data/test_classes.csv')
        self.conf_matrix = list()
        self.heat_map = None
        self.ROC = None
        self.model = model

    def save(self):
        self.model.save(self.model_path)
        self.model = None
        with open(self.pickle_path, 'wb') as fb:
            pickle.dump(self, fb)
        self.model = load_model(self.model_path)

        
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
        
        # Using tensorboard to make graphs for the scoring report, and for choosing best model
        os.mkdir('temp_log')
        os.mkdir('temp_model')
        tensorboard_callback = TensorBoard(
                    log_dir=f"temp_log",
                    histogram_freq=0,
                    write_graph=False,
                    write_images=False,
                    update_freq='epoch',
                    profile_batch=2,
                    embeddings_freq=0,
                    embeddings_metadata=None)
        tensor_checkpoint = ModelCheckpoint('temp_model', save_best_only=True)

        # Filter needs to be a PIL.Image filter
        for filter in filters:
            if filter in ('BLUR','CONTOUR','DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE',
                            'EMBOSS','FIND_EDGES','SHARPEN','SMOOTH','SMOOTH_MORE','train'):

                # Keeping track of filters used in preprocessing
                self.filters.append(filter)
                
                # Data generators make training the model easy and fun!
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

                # Time to fit our generator with the training images!
                self.model.fit_generator(
                            train_generator,
                            epochs=epochs,
                            validation_data= validation_generator,
                            steps_per_epoch=450,
                            validation_steps=150,
                            shuffle=True,
                            workers=0,
                            callbacks=[tensorboard_callback,tensor_checkpoint])

        # Creating a graph of loss and accuracy from the Tensorboard log files
        fig, axes = plt.subplots(2,1, sharex=True)

        # Plotting train and val data
        for d in ['train','validation']:
            ea = event_accumulator.EventAccumulator(f'temp_log/{d}')
            df = pd.DataFrame(ea.Reload().Scalars('epoch_loss'))
            axes[0].plot(df.step, df.value, label=d)
            df = pd.DataFrame(ea.Reload().Scalars('epoch_accuracy'))
            axes[1].plot(df.step, df.value, label=d)

        # Setting labels and titles
        axes[1].set_xlabel('Epochs')
        axes[1].legend()
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel('Accuracy')
        fig.suptitle(f'{self.name}')

        # Save graph and delete temp files
        self.graph = fig
        shutil.rmtree('temp_log')

        # Save the best version of this model and delete temp files
        self.model = load_model('temp_model')
        shutil.rmtree('temp_model')

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

        def predict(x):
            image = Image.open(f'data/test/{x.Filename}')
            image = np.asarray(image.resize((30,30)))
            image = np.resize(image, [1,30,30,3])
            certainty_values = self.model.predict(image)[0]
            predicted_class = np.argmax(certainty_values, axis=-1)
            return pd.Series({
                'image_location':x.Filename,
                'actual_class':x.ClassId,
                'predicted_class':predicted_class,
                'confidence':certainty_values[predicted_class]})

        self.preds = self.preds.apply(predict, axis=1)

        # Basic model information will be saved to index.json for viewing on home page
        report = {'id' : self.index}
        report['name'] = self.name
        report['timestamp'] = self.timestamp
        report['model path'] = self.model_path
        report['pickle path'] = self.pickle_path
                
        # Add basic stats and scores of the model to the report
        acc_preds = self.preds[self.preds['predicted_class'] == self.preds['actual_class']]
        report['True Predictions'] = len(acc_preds)
        report['Total Predictions'] = len(acc_preds)
        report['Accuracy score'] = f'{round((len(acc_preds)/len(self.preds))*100)}%'

        # Save the report summary into index.txt
        with open('models/index.txt', 'a') as fp:
            print(report, file=fp)

        # Creating confusion matrix and storing it in self.conf_matrix
        self.conf_matrix = list()
        for sign in range(43):
            ls = [0]*43
            sign_preds = self.preds[self.preds.actual_class == sign]
            group_preds = sign_preds.groupby('predicted_class').count()
            for i in group_preds.index:
                ls[i] = group_preds.loc[i]['actual_class']
            self.conf_matrix.append(ls)

        # Create heat map with seaborn
        fig, ax = plt.subplots()
        ax = sns.heatmap(self.conf_matrix, vmin=0, vmax=100)
        fig.suptitle('Confusion Matrix')
        self.heat_map = fig

        # Create modified ROC Curve plot
        fig, axes = plt.subplots(2,1, sharex=True)

        x = list()
        tp_y = list()
        pd_y = list()

        # Creating a modified ROC curve
        for threshold in range(100):

            # All predictions that are above the confidence threshold (predictions above threshold)
            pat = self.preds[self.preds.confidence >= (threshold/100)]
            tp_rate = len(pat[pat['predicted_class'] == pat['actual_class']]) / len(pat)
            drop_rate = 1 -(len(pat) / len(self.preds))
            x.append(threshold)
            tp_y.append(tp_rate)
            pd_y.append(drop_rate)

        axes[0].plot(x, tp_y)
        axes[1].plot(x, pd_y)

        axes[1].set_xlabel('Confidence Threshold')
        axes[0].set_ylabel('True Positive Rate')
        axes[1].set_ylabel('Unclassified Image Rate')
        fig.suptitle('Modified ROC Curve')
        self.ROC = fig

        self.save()

if __name__ == '__main__':
    if not os._exists('models/index.txt'): pass
    else: os.mkdir('models/index.txt')
    model = Model('traffic')
    model.add_layers(depth=2)
    model.fit(epochs=15)
    model.score()



# filters=['BLUR','CONTOUR','DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE',
#                             'EMBOSS','FIND_EDGES','SHARPEN','SMOOTH','SMOOTH_MORE','train']
    