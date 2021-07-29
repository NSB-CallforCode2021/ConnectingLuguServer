'''
===============================================================================
# ricehealthcheck
A supervision model based on mobilenet to check rice health by image.
===============================================================================
'''


import os
import cv2
import shutil
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import albumentations as albu
from sklearn.utils import shuffle
from tensorflow.keras import models
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, CSVLogger, LearningRateScheduler)

aug_types = albu.Compose([
    albu.HorizontalFlip(),
    albu.OneOf([
        albu.HorizontalFlip(),
        albu.VerticalFlip(),
    ], p=0.8),
    albu.OneOf([
        albu.RandomContrast(),
        albu.RandomGamma(),
        albu.RandomBrightness(),
    ], p=0.3),
    albu.OneOf([
        albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        albu.GridDistortion(),
        albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.3),
    albu.ShiftScaleRotate()
])

class RiceHealthModel():
    def __init__(self, image_hight=224, image_width=224, image_channels=3):
        self.IMAGE_HEIGHT = image_hight
        self.IMAGE_WIDTH = image_width
        self.IMAGE_CHANNELS = image_channels
        self.LABELS = ['bacterial leaf blight', 'brown spot', 'leaf smut', "healthy rice"]
        self.cols = ['target_bacterial_leaf_blight', 'target_brown_spot', 'target_leaf_smut', 'target_healthy_rice']
        self._score = [70, 80, 90, 100]
        self._model = None
        self._model_path = "./model_dir/leafDiseaseAnalyzer.h5"
        self._image_dir = './image_dir'

    def train(self):
        self.prepare_data()
        # train_gen = self.train_generator(batch_size=8)
        # val_gen = self.val_generator(batch_size=5)
        # test_gen = self.test_generator(batch_size=1)
        model = MobileNet(weights='imagenet')

        # Exclude the last 2 layers of the above model.
        x = model.layers[-2].output

        # Create a new dense layer for predictions
        # 3 corresponds to the number of classes
        predictions = Dense(4, activation='softmax')(x)

        # inputs=model.input selects the input layer, outputs=predictions refers to the
        # dense layer we created above.

        model = Model(inputs=model.input, outputs=predictions)

        model.summary()
        TRAIN_BATCH_SIZE = 8
        VAL_BATCH_SIZE = 5

        num_train_samples = len(self.df_train)
        num_val_samples = len(self.df_val)
        train_batch_size = TRAIN_BATCH_SIZE
        val_batch_size = VAL_BATCH_SIZE

        train_steps = np.ceil(num_train_samples / train_batch_size)
        train_gen = self.train_generator(batch_size=TRAIN_BATCH_SIZE)

        val_steps = np.ceil(num_val_samples / val_batch_size)
        val_gen = self.val_generator(batch_size=VAL_BATCH_SIZE)
        model.compile(
            Adam(lr=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        filepath = "model.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')
        log_fname = 'training_log.csv'
        csv_logger = CSVLogger(filename=log_fname,
                               separator=',',
                               append=False)

        callbacks_list = [checkpoint, csv_logger]

        history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=100,
                                      validation_data=val_gen, validation_steps=val_steps,
                                      verbose=1,
                                      callbacks=callbacks_list)
        train_log = pd.read_csv('training_log.csv')
        model.load_weights('model.h5')
        model.save('leafDiseaseAnalyzer.h5')
        val_gen = self.val_generator(batch_size=1)

        val_loss, val_acc = \
            model.evaluate_generator(val_gen,
                                     steps=len(self.df_val))
        print('val_loss:', val_loss)
        print('val_acc:', val_acc)
        import matplotlib.pyplot as plt

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.figure()


        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()

        plt.show()
        test_gen = self.test_generator(batch_size=1)
        preds = model.predict_generator(test_gen, steps=len(self.df_val), verbose=1)
        y_pred = np.argmax(preds, axis=1)
        y_true = self.df_val[self.cols]
        y_true = np.asarray(y_true)
        y_true = np.argmax(y_true, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        cm_plot_labels = ['leaf_blight', 'brown_spot', 'leaf_smut', "healthy_rice"]

        plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
        report = classification_report(y_true, y_pred, target_names=['bacterial_leaf_blight', 'brown_spot', 'leaf_smut', "healthy_rice"])

        print(report)

    def prepare_data(self):
        leaf_smut_list = \
            os.listdir('../data/ricedata/Leaf smut')
        brown_spot_list = \
            os.listdir('../data/ricedata/Brown spot')
        bacterial_leaf_blight_list = \
            os.listdir('../data/ricedata/Bacterial leaf blight')
        healthy_list = \
            os.listdir('../data/ricedata/Healthy rice')
        df_leaf_smut = pd.DataFrame(leaf_smut_list, columns=['image'])
        df_leaf_smut['target'] = 'leaf_smut'

        df_brown_spot = pd.DataFrame(brown_spot_list, columns=['image'])
        df_brown_spot['target'] = 'brown_spot'

        df_bacterial_leaf_blight = pd.DataFrame(bacterial_leaf_blight_list, columns=['image'])
        df_bacterial_leaf_blight['target'] = 'bacterial_leaf_blight'

        df_healthy_rice = pd.DataFrame(healthy_list, columns=['image'])
        df_healthy_rice['target'] = 'healthy_rice'

        # Sample 5 validation images from each class
        df_leaf_smut_val = df_leaf_smut.sample(n=5, random_state=101)
        df_brown_spot_val = df_brown_spot.sample(n=5, random_state=101)
        df_bacterial_leaf_blight_val = df_bacterial_leaf_blight.sample(n=5, random_state=101)
        df_healthy_rice_val = df_healthy_rice.sample(n=2, random_state=101)

        # leaf_smut
        val_list = list(df_leaf_smut_val['image'])
        # filter out the val images
        df_leaf_smut_train = df_leaf_smut[~df_leaf_smut['image'].isin(val_list)] # ~ means notin


        val_list = list(df_brown_spot_val['image'])
        df_brown_spot_train = df_brown_spot[~df_brown_spot['image'].isin(val_list)] # ~ means notin

        val_list = list(df_bacterial_leaf_blight_val['image'])
        df_bacterial_leaf_blight_train = \
            df_bacterial_leaf_blight[~df_bacterial_leaf_blight['image'].isin(val_list)] # ~ means notin

        val_list = list(df_healthy_rice_val['image'])
        df_healthy_rice_train = \
            df_healthy_rice[~df_healthy_rice['image'].isin(val_list)]

        df_data = pd.concat([df_leaf_smut, df_brown_spot, df_bacterial_leaf_blight, df_healthy_rice], axis=0).reset_index(drop=True)

        df_train = \
            pd.concat([df_leaf_smut_train, df_brown_spot_train, df_bacterial_leaf_blight_train, df_healthy_rice_train], axis=0).reset_index(drop=True)

        df_val = \
            pd.concat([df_leaf_smut_val, df_brown_spot_val, df_bacterial_leaf_blight_val, df_healthy_rice_val], axis=0).reset_index(drop=True)

        df_data = shuffle(df_data)
        df_train = shuffle(df_train)
        df_val = shuffle(df_val)
        df_data['target'].value_counts()
        df_train['target'].value_counts()
        df_val['target'].value_counts()
        val_len = len(df_val)
        train_len = len(df_train)
        df_combined =  pd.concat(objs=[df_val, df_train], axis=0).reset_index(drop=True)
        # create the dummy variables
        df_combined = pd.get_dummies(df_combined, columns=['target'])

        # separate the train and val sets
        df_val = df_combined[:val_len]
        df_train = df_combined[val_len:]
        df_combined.to_csv('df_combined.csv.gz', compression='gzip', index=False)
        df_train.to_csv('df_train.csv.gz', compression='gzip', index=False)
        df_val.to_csv('df_val.csv.gz', compression='gzip', index=False)
        image_dir = 'image_dir'
        os.mkdir(image_dir)
        leaf_smut_list = \
            os.listdir('../data/ricedata/Leaf smut')
        brown_spot_list = \
            os.listdir('../data/ricedata/Brown spot')
        bacterial_leaf_blight_list = \
            os.listdir('../data/ricedata/Bacterial leaf blight')
        healthy_rice_list = os.listdir('../data/ricedata/Healthy rice')

        for path, sample_list in zip(('../data/ricedata/Leaf smut/', '../data/ricedata/Brown spot/', \
                     '../data/ricedata/Bacterial leaf blight/', '../data/ricedata/Healthy rice/'),\
                        (leaf_smut_list, brown_spot_list, bacterial_leaf_blight_list, \
                         healthy_rice_list)):
            for fname in sample_list:
                self._cp_file(path, fname)

        self.df_train = df_train
        self.df_val = df_val

    def train_generator(self, batch_size=8):

        while True:

            # load the data in chunks (batches)
            for df in pd.read_csv('df_train.csv.gz', chunksize=batch_size):
                image_id_list = list(df['image'])

                # Create empty X matrix - 3 channels
                X_train = np.zeros((len(df), self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS), dtype=np.uint8)
                for i in range(0, len(image_id_list)):
                    # get the image and mask
                    image_id = image_id_list[i]

                    # set the path to the image
                    path = 'image_dir/' + image_id

                    # read the image
                    image = cv2.imread(path)
                    # convert to from BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # resize the image
                    image = cv2.resize(image, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))

                    y_train = df[self.cols]
                    y_train = np.asarray(y_train)

                    aug_image = augment_image(aug_types, image)

                    # insert the image into X_train
                    X_train[i] = aug_image

                # Normalize the images
                X_train = X_train/255

                yield X_train, y_train

    def val_generator(self, batch_size=5):
        while True:

            # load the data in chunks (batches)
            for df in pd.read_csv('df_val.csv.gz', chunksize=batch_size):

                # get the list of images
                image_id_list = list(df['image'])

                # Create empty X matrix - 3 channels
                X_val = np.zeros((len(df), self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS), dtype=np.uint8)

                for i in range(0, len(image_id_list)):
                    # get the image and mask
                    image_id = image_id_list[i]


                    # set the path to the image
                    path = 'image_dir/' + image_id

                    # read the image
                    image = cv2.imread(path)
                    # convert to from BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # resize the image
                    image = cv2.resize(image, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))

                    # insert the image into X_train
                    X_val[i] = image

                    y_val = df[self.cols]
                    y_val = np.asarray(y_val)

                # Normalize the images
                X_val = X_val/255

                yield X_val, y_val

    def test_generator(self, batch_size=1):

        while True:

            # load the data in chunks (batches)
            for df in pd.read_csv('df_val.csv.gz', chunksize=batch_size):

                # get the list of images
                image_id_list = list(df['image'])

                # Create empty X matrix - 3 channels
                X_test = np.zeros((len(df), self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS), dtype=np.uint8)

                for i in range(0, len(image_id_list)):


                    # get the image and mask
                    image_id = image_id_list[i]


                    # set the path to the image
                    path = 'image_dir/' + image_id

                    # read the image
                    image = cv2.imread(path)

                    # convert to from BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # resize the image
                    image = cv2.resize(image, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))

                    # insert the image into X_train
                    X_test[i] = image
                # Normalize the images
                X_test = X_test/255
                yield X_test


    def _cp_file(self, path, fname):
        # source path to image
        src = os.path.join(path, fname)
        # destination path to image
        dst = os.path.join(self._image_dir, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)

    def predict(self, image_path):
        # check whether image is legal
        if(image_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', \
                                        '.pgm', '.ppm', '.tif', '.tiff')) is not True):
            return {"error": "Illegal image to do analysis."}
        # check whetehr image exists
        if (os.path.isfile(image_path) is not True):
            return {"error": "Image not exists."}

        img_arr = np.zeros((1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS), dtype=np.uint8)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.IMAGE_HEIGHT, self.IMAGE_WIDTH))
        img_arr[0] = img
        # normalize input pixel to [0, 1]
        img_arr = img_arr / 255
        if self._model is None:
            self._load_model(self._model_path)
        preds = self._model.predict(img_arr)
        preds_idx = np.argmax(preds, axis=1)
        return {"score": self._score[preds_idx[0]], "disease": self.LABELS[preds_idx[0]]}


    def _load_model(self, model_path):
        self._model = models.load_model(model_path)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def augment_image(augmentation, image):

    """
    Uses the Albumentations library.

    Inputs:
    1. augmentation - this is the instance of type of augmentation to do
    e.g. aug_type = HorizontalFlip(p=1)
    # p=1 is the probability of the transform being executed.

    2. image - image with shape (h,w)

    Output:
    Augmented image as a numpy array.

    """
    # get the transform as a dict
    aug_image_dict =  augmentation(image=image)
    # retrieve the augmented matrix of the image
    image_matrix = aug_image_dict['image']


    return image_matrix

if __name__ == "__main__":
    RiceHealthModel().train()
    # print(RiceHealthModel().predict("/home/zewenh/git/CVLED/riceanalysis/image_dir/DSC_0101.jpg"))
