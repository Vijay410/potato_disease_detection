import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from settings import *

class LoadByGenerator():
    ##
    '''
    Data Augmentation:
    One way to fix overfitting is to augment the dataset so that it has a sufficient number of training examples.
    Data augmentation takes the approach of generating more training data from existing training samples by
    augmenting the samples using random transformations that yield believable-looking images.
    The goal is the model will never see the exact same picture twice during training. This helps expose the model to more aspects of the data and generalize better.
    '''

    def __init__(self, path='DIRECTORY_DATASET'):
        self._image_gen_train = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=45, # rotation
            width_shift_range=.15,
            height_shift_range=.15,
            horizontal_flip=True, # apply horizontal_flip
            zoom_range=0.5 # apply zoom
        )
        self._path = path

    def load_dataset(self):
        # Splitting the data into training and validation sets
        train_dir = os.path.join(self._path, 'train')
        validation_dir = os.path.join(self._path, 'validation')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)
        
        # Loop through each class folder
        for class_folder in os.listdir(self._path):
            if os.path.isdir(os.path.join(self._path, class_folder)):
                class_images = os.listdir(os.path.join(self._path, class_folder))
                train_images, validation_images = train_test_split(class_images, test_size=0.2, random_state=42)
                
                # Move images to respective directories
                for image in train_images:
                    shutil.move(os.path.join(self._path, class_folder, image), os.path.join(train_dir, class_folder))
                
                for image in validation_images:
                    shutil.move(os.path.join(self._path, class_folder, image), os.path.join(validation_dir, class_folder))
        
        train_data_gen = self._image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=train_dir,
                                                                  shuffle=True,
                                                                  target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                                  class_mode='binary')
        validation_data_gen = self._image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                                  directory=validation_dir,
                                                                  shuffle=True,
                                                                  target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                                  class_mode='binary')
        return train_data_gen, validation_data_gen

if __name__== "__main__":
    loader=LoadByGenerator('../dataset/dataset_separated')
    training_set, validation_set= loader.load_dataset()
    print(training_set)
