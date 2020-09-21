from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
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
    def __init__(self, name, description=''):
        self.index = len(os.listdir('models/'))
        self.name = name
        self.timestamp = datetime.now().strftime('%m/%d %H:%M')
        self.model_path = f'models/{self.index}_{self.name}'
        self.pickle_path = f'pickles/{self.index}_{self.name}.pk'
        self.report_image_path = f'app/static/report_images/{self.index}_{self.name}_'
        self.description = description
        self.filters = list()
        self.architecture = list()
        self.summary = list()
        self.preds = pd.read_csv('data/test_classes.csv')
        self.conf_matrix = list()
        self.fit_time = 0
        self.class_data = pd.DataFrame(columns=['most_tp', 'least_tp', 'acc', 'class_pred'])

    def save(self, keras_model):
        keras_model.save(self.model_path)
        with open(self.pickle_path, 'wb') as fb:
            pickle.dump(self, fb)
        
    def add_layers(self, depth=3, dropout=.25, func='relu', normalize=False, learning_rate=.001):
        '''Add 2,4,or 6 layers to your network.
        Choose a dropout rate, and an activation function too!'''
        model = Sequential()
        self.architecture = [depth, dropout, func]
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation=func, input_shape=(30, 30,3)))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation=func))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(rate=dropout))
        if normalize: model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation=func))
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation=func))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(rate=dropout))
        if normalize: model.add(BatchNormalization())
        # model.add(Conv2D(filters=64, kernel_size=(3,3), activation=func))
        # model.add(Conv2D(filters=64, kernel_size=(3,3), activation=func))
        # model.add(MaxPool2D(pool_size=(2,2)))
        # model.add(Dropout(rate=dropout))
        # model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(256, activation=func))
        model.add(Dropout(rate=dropout))
        model.add(Dense(43, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy']
            )
        model.summary(print_fn= lambda x: self.summary.append(x))
        self.save(model)

    def fit(self, filters=['train'], epochs=10):
        '''Fit the model with a selection of image preprocessing filter
        and number of epochs'''
        model = load_model(self.model_path)
        os.mkdir('temp_model')

        # Using tensorboard to make graphs for the scoring report, and for choosing best model
        tensorboard_callback = TensorBoard(
                    log_dir=f"temp_log",
                    histogram_freq=0,
                    write_graph=False,
                    write_images=False,
                    update_freq='epoch',
                    profile_batch=2,
                    embeddings_freq=0,
                    embeddings_metadata=None)
        tensor_checkpoint = ModelCheckpoint('temp_model', save_best_only=False)

        # Creating a graph of loss and accuracy from the Tensorboard log files
        fig, axes = plt.subplots(2,1, sharex=True)

        # Filter needs to be a PIL.Image filter
        for filter in filters:
            if filter in ('BLUR','CONTOUR','DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE',
                            'EMBOSS','FIND_EDGES','SHARPEN','SMOOTH','SMOOTH_MORE','train', 
                            'contrast', 'brightness'):
                self.filters.append(filter)
                os.mkdir('temp_log')
                
                
                # Data generators make training the model easy and fun!
                datagen = ImageDataGenerator(
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

                # Timing the fit for report, only timing the fit
                start = time.time()

                # Time to fit our generator with the training images!
                model.fit_generator(
                            train_generator,
                            epochs=epochs,
                            validation_data= validation_generator,
                            steps_per_epoch=450,
                            validation_steps=150,
                            shuffle=True,
                            workers=0,
                            callbacks=[tensorboard_callback,tensor_checkpoint])

                # Finish timing the fit time for report
                self.fit_time += int((time.time() - start)/60)

                # Plotting train and val logs 
                # Must be done for each filter otherwise the graph connects the start/ends
                for d in ['train','validation']:
                    ea = event_accumulator.EventAccumulator(f'temp_log/{d}')
                    df = pd.DataFrame(ea.Reload().Scalars('epoch_loss'))
                    axes[0].plot(df.step, df.value, label=f'{d}_{filter}')
                    df = pd.DataFrame(ea.Reload().Scalars('epoch_accuracy'))
                    axes[1].plot(df.step, df.value, label=f'{d}_{filter}')  

                # Removing the logs and using the best model after each filter training
                # ensures the best model from each round and no double graphing of data
                # If a better model was not made with this filter, then pass
                shutil.rmtree('temp_log')

        # Setting labels and titles
        axes[1].set_xlabel('Epochs')
        axes[1].legend(loc='upper left', bbox_to_anchor=(1,2), fancybox=True)
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel('Accuracy')
        fig.suptitle(f'Loss and Accuracy logged during training.')

        # Save graph and delete temp files
        fig.savefig(f'{self.report_image_path}graph.png', bbox_inches='tight')
        plt.close('all')
        
        # Save the best version of this model and delete temp files
        self.save(model)
        shutil.rmtree('temp_model')

    def score(self):
        '''[score_it] function, for evaluating saved models with test images.

        Requires a model name (string) to designate which saved model to use.
        Saved model must reside in models directory in a folder with the 
        same name. Use Keras.models.save_model(model, f'models/{model_name})
        
        Reports are written to /reports/{model_name}/ directory. If previously
        made reports exist, then report directory is numbered like {model_name1}
        
        There is no need to designate test data. This class will use the test
        set located in data/test.'''

        model = load_model(self.model_path)

        def predict(x):
            image = Image.open(f'data/test/{x.Filename}')
            image = np.asarray(image.resize((30,30)))
            image = np.resize(image, [1,30,30,3])
            certainty_values = model.predict(image)[0]
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
        report['description'] = self.description
        report['timestamp'] = self.timestamp
        report['model path'] = self.model_path
        report['pickle path'] = self.pickle_path
                
        # Add basic stats and scores of the model to the report
        acc_preds = self.preds[self.preds['predicted_class'] == self.preds['actual_class']]
        report['True Predictions'] = len(acc_preds)
        report['Total Predictions'] = len(self.preds)
        report['Accuracy score'] = f'{round(len(acc_preds)/len(self.preds), 4)*100}%'

        # Save the report summary into index.txt
        with open('models/index.txt', 'a') as fp:
            print(report, file=fp)


        # Creating class specific analysis data
        for sign in range(43):
        
            # DFs to use for class specific analysis
            sign_preds = self.preds[self.preds.actual_class == sign]
            group_preds = sign_preds.groupby('predicted_class').count()

            # Confusion Matrix data is created here, as well as class predictions for report
            conf_matrix_line = [0]*43
            class_pred = list()

            # Loop through every class that this class was predicted as
            # Confusion matrix uses the %, while group preds for report are actual numbers
            for i in group_preds.index:
                conf_matrix_line[i] = group_preds.loc[i]['actual_class'] / group_preds['actual_class'].sum()
                class_pred.append((i, group_preds.loc[i]['actual_class']))
            self.conf_matrix.append(conf_matrix_line)
            

            # Recording the most certain image classified, and least certain, for report
            true_preds = sign_preds[sign_preds.predicted_class == sign_preds.actual_class]
            if len(true_preds) > 2:
                most_tp = true_preds.iloc[true_preds.confidence.argmax()]['image_location']
                least_tp = true_preds.iloc[true_preds.confidence.argmin()]['image_location']
            else:
                most_tp = None
                least_tp = None
            

            # Accuracy of the model on this class
            acc = round(len(true_preds)/len(sign_preds), 4) * 100

            # Class specific data is recorded in a df
            self.class_data = self.class_data.append(
                {'most_tp':most_tp,
                'least_tp':least_tp,
                'acc':acc,
                'class_pred':class_pred},
                ignore_index=True)

        # Create heat map with seaborn
        fig, ax = plt.subplots()
        ax = sns.heatmap(self.conf_matrix, vmin=0, vmax=1)
        fig.suptitle('Confusion Matrix')
        fig.savefig(f'{self.report_image_path}heat_map.png')
        plt.close('all')

        # Create ROC Curve plot
        fig, axes = plt.subplots(2,1, sharex=True)


        x = list()
        tp_y = list()
        pd_y = list()

        # Creating an ROC curve
        for threshold in range(100):

            # All predictions that are above the confidence threshold (predictions above threshold)
            classified = self.preds[self.preds.confidence > threshold/100]
            true_positives = classified[classified.predicted_class == classified.actual_class]
            unclassified = self.preds[self.preds.confidence <= (threshold/100)]
            true_negatives = unclassified[unclassified.predicted_class != unclassified.actual_class]
            tp_rate = (len(true_positives) + len(true_negatives)) / len(self.preds)

            drop_rate = 1 -(len(unclassified) / len(self.preds))
            x.append(threshold)
            tp_y.append(tp_rate)
            pd_y.append(drop_rate)

        axes[0].plot(x, tp_y)
        axes[1].plot(x, pd_y)

        axes[1].set_xlabel('Confidence Threshold')
        axes[0].set_ylabel('True Positive Rate')
        axes[1].set_ylabel('Classified Image Rate')

        fig.suptitle('Confidenct Threshold Curve')
        fig.savefig(f'{self.report_image_path}ROC.png')
        plt.close()

        self.save(model)

if __name__ == '__main__':
    if not os._exists('models/index.txt'): pass
    else: os.mkdir('models/index.txt')

    # modelA = Model('traffic', 'A simple model from data-flair.training/blogs/python-project-traffic-signs-recognition')
    # modelA.add_layers(dropout=.5, func='relu')
    # modelA.fit(filters=['train'], epochs=15)
    # modelA.score()
    # del modelA

    # print('modelB')
    # modelB = Model('traffic', 'Low dropout and learning rate, adding batch normalization to prevent over fitting')
    # modelB.add_layers(dropout=.1, func='relu', normalize=True, learning_rate=.0001)
    # modelB.fit(filters=['train'], epochs=15)
    # modelB.score()
    # del modelB

    # print('modelC')
    # modelC = Model('traffic', 'Swich act. func., plus brightness, sharpen, and smooth preprocessing. Epochs to 25')
    # modelC.add_layers(dropout=.1, func='swish', normalize=True, learning_rate=.0001)
    # modelC.fit(filters=['train', 'brightness', 'SHARPEN','SMOOTH'], epochs=25)
    # modelC.score()
    # del modelC
    
    # print('modelD')
    # modelD = Model('traffic', 'All filters and 75 epochs')
    # modelD.add_layers(dropout=.1, func='swish', normalize=True, learning_rate=.0001)
    # modelD.fit(filters=['train','brightness','BLUR','CONTOUR','DETAIL','DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE','EDGE_ENHANCE_MORE','train','brightness', 'EMBOSS','FIND_EDGES','FIND_EDGES','SHARPEN','SHARPEN','SMOOTH','SMOOTH_MORE','train', 'brightness'], epochs=75)
    # modelD.score()
    # del modelD

    # modelB = Model('traffic', 'Adding batch normalization,  selu ')
    # modelB.add_layers(dropout=.3, func='relu', normalize=True)
    # modelB.fit(filters=['train'], epochs=15)
    # modelB.score()
    # del modelB

    # modelC = Model('traffic', 'Trying out preprocessing filters')
    # modelC.add_layers(dropout=.3, func='relu', normalize=True)
    # modelC.fit(filters=['train', 'BLUR', 'CONTOUR','DETAIL','EDGE_ENHANCE'], epochs=15)
    # modelC.score()
    # del modelC

    # modelD = Model('traffic', 'learning rate to .0001, dropout to .2, different filters, epochs to 25')
    # modelD.add_layers(dropout=.2, func='relu', normalize=True, learning_rate=.0005)
    # modelD.fit(filters=['train', 'brightness', 'EDGE_ENHANCE_MORE' ], epochs=50)
    # modelD.score()


    # modelD = Model('traffic', 'Trying out swish AF')
    # modelD.add_layers(dropout=.3, func='swish', normalize=True, learning_rate=.001)
    # modelD.fit(filters=['train' ], epochs=15)
    # modelD.score()
    # del modelD

    

    # modelD = Model('traffic', 'Three train set fittings')
    # modelD.add_layers(dropout=.2, func='relu', normalize=True, learning_rate=.001)
    # modelD.fit(filters=['train', 'train', 'train'], epochs=25)
    # modelD.score()
    # del modelD


    # modelD = Model('traffic', 'Low learning rate but high epochs')
    # modelD.add_layers(dropout=.2, func='relu', normalize=True, learning_rate=.00005)
    # modelD.fit(filters=['train', 'brightness', 'EDGE_ENHANCE_MORE' ], epochs=75)
    # modelD.score()
    # del modelD

    # modelD = Model('traffic', 'Only smooting filters')
    # modelD.add_layers(dropout=.3, func='relu', normalize=True, learning_rate=.001)
    # modelD.fit(filters=['train', 'SMOOTH','SMOOTH_MORE','BLUR' ], epochs=25)
    # modelD.score()
    # del modelD

    # modelD = Model('traffic', 'Only sharpening filters')
    # modelD.add_layers(dropout=.3, func='relu', normalize=True, learning_rate=.001)
    # modelD.fit(filters=['train', 'DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE','FIND_EDGES','SHARPEN' ], epochs=50)
    # modelD.score()
    # del modelD

    # modelD = Model('traffic', 'High dropout with normalize and 50 epochs')
    # modelD.add_layers(dropout=.5, func='relu', normalize=True, learning_rate=.001)
    # modelD.fit(filters=['train', 'brightness', 'EDGE_ENHANCE_MORE' ], epochs=50)
    # modelD.score()

    # modelD = Model('traffic', '')
    # modelD.add_layers(dropout=.3, func='swish', normalize=True, learning_rate=.0001)
    # modelD.fit(filters=['BLUR','CONTOUR','DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE', 'EMBOSS','FIND_EDGES','SHARPEN','SMOOTH','SMOOTH_MORE','train' ], epochs=75)
    # modelD.score()

    # modelD = Model('traffic', 'Sharp filters twice')
    # modelD.add_layers(dropout=.3, func='swish', normalize=True, learning_rate=.00001)
    # modelD.fit(filters=['train','brightness','BLUR','CONTOUR','DETAIL','DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE','EDGE_ENHANCE_MORE','train','brightness', 'EMBOSS','FIND_EDGES','FIND_EDGES','SHARPEN','SHARPEN','SMOOTH','SMOOTH_MORE','train', 'brightness'], epochs=75)
    # modelD.score()

    # modelD = Model('traffic', 'multiple filter runs')
    # modelD.add_layers(dropout=.3, func='swish', normalize=True, learning_rate=.001)
    # modelD.fit(filters=['train'], epochs=10)
    # modelD.score()

        
    print('modelD')
    modelD = Model('traffic', 'All filters and 75 epochs, very low learning rate')
    modelD.add_layers(dropout=.1, func='swish', normalize=True, learning_rate=.00001)
    modelD.fit(filters=['train','brightness','BLUR','CONTOUR','DETAIL','DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE','EDGE_ENHANCE_MORE','train','brightness', 'EMBOSS','FIND_EDGES','FIND_EDGES','SHARPEN','SHARPEN','SMOOTH','SMOOTH_MORE','train', 'brightness'], epochs=75)
    modelD.score()
    del modelD

    plt.close('all')

# filters=['BLUR','CONTOUR','DETAIL','EDGE_ENHANCE','EDGE_ENHANCE_MORE',
#                             'EMBOSS','FIND_EDGES','SHARPEN','SMOOTH','SMOOTH_MORE','train']
    