import model_it

class Traffic_Control(model_it):
    def __init__(self):
        pass

    def _model_it(img_dir):

        shutil.rmtree(f'logs/{img_dir}/')

        datagen = ImageDataGenerator(rescale=1./255, #scales image data to be between 0 and 1, why? I don't know!
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        rotation_range=45,
                                        horizontal_flip=False,
                                        vertical_flip=False,
                                        validation_split = .25)

        train_generator = datagen.flow_from_directory(
                    f'data/{img_dir}/', 
                    target_size=(30, 30),
                    batch_size=64,
                    subset='training')

        validation_generator = datagen.flow_from_directory(
                    f'data/{img_dir}/',
                    target_size=(30, 30),
                    batch_size=64,
                    subset='validation')

        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(30, 30,3)))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(rate=.25))
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(rate=.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=.25))
        model.add(Dense(43, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tensorboard_callback = TensorBoard(log_dir=f'logs/{img_dir}/', histogram_freq=0, write_graph=True, write_images=False,
            update_freq='epoch', profile_batch=2, embeddings_freq=0,
            embeddings_metadata=None, )

        tensor_checkpoint = ModelCheckpoint(f'models/{img_dir}/', save_best_only=True)

        model.fit_generator(
                    train_generator,
                    epochs=10,
                    validation_data= validation_generator,
                    steps_per_epoch=450,
                    validation_steps=150,
                    shuffle=True,
                    callbacks=[
                        tensorboard_callback,
                        tensor_checkpoint],
                    workers=0)

    def score_it(model):

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
        print(f'Accuracy = {len(falsities)/len(predictions)}')
