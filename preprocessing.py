import tensorflow as tf
from tensorflow.keras import layers
from logging_config import setup_custom_logger

class ImageDataPipeline:

    def __init__(self, directory, batch_size=64, image_size=256, channels=3, epochs=50):
        self.directory = directory
        self.batch_size = batch_size
        self.image_size = (image_size, image_size)
        self.channels = channels
        self.epochs = epochs
        self.SEED = 123  # Define SEED
        self.SHUFFLE_SIZE = 1000  # Define SHUFFLE_SIZE
        self.BUFFER_SIZE = tf.data.AUTOTUNE  # Define BUFFER_SIZE
        self.logger = setup_custom_logger(__name__)
        

    def load_dataset(self):
        try:
            dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.directory,
                seed=self.SEED,
                shuffle=True,
                image_size=self.image_size,
                batch_size=self.batch_size
            )
            return dataset
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
        
    def partition_dataset(self, dataset, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
        try:
            assert (train_split + test_split + val_split) == 1

            ds_size = len(dataset)

            if shuffle:
                dataset = dataset.shuffle(shuffle_size, seed=12)

            train_size = int(train_split * ds_size)
            val_size = int(val_split * ds_size)

            train_ds = dataset.take(train_size)
            val_ds = dataset.skip(train_size).take(val_size)
            test_ds = dataset.skip(train_size).skip(val_size)

            return train_ds, val_ds, test_ds
        except Exception as e:
            self.logger.error(f"Error partitioning dataset: {e}")
            raise

    def cache_shuffle_prefetch(self, dataset):
        try:
            return dataset.cache().shuffle(self.SHUFFLE_SIZE).prefetch(buffer_size=self.BUFFER_SIZE)
        except Exception as e:
            self.logger.error(f"Error caching, shuffling, and prefetching dataset: {e}")
            raise

    def resize_and_rescale_model(self):
        try:
            resize_and_rescale = tf.keras.Sequential([
                layers.Resizing(self.image_size[0], self.image_size[1]),
                layers.Rescaling(1./255),
            ])
            return resize_and_rescale
        except Exception as e:
            self.logger.error(f"Error creating resize and rescale model: {e}")
            raise

    def data_augmentation_model(self):
        try:
            return tf.keras.Sequential([
                layers.RandomFlip("horizontal_and_vertical"),
                layers.RandomRotation(0.2),
            ])
        except Exception as e:
            self.logger.error(f"Error creating data augmentation model: {e}")
            raise

    def apply_data_augmentation(self, dataset, data_augmentation):
        try:
            return dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y)
            ).prefetch(buffer_size=self.BUFFER_SIZE)
        except Exception as e:
            self.logger.error(f"Error applying data augmentation: {e}")
            raise
