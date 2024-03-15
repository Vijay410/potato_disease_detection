import tensorflow as tf
from tensorflow.keras import models, layers
from logging_config import setup_custom_logger

class ModelBuilder:
    logger = setup_custom_logger(__name__)

    @staticmethod
    def build_model(input_shape, num_classes, resize_and_rescale):
        try:
            model = models.Sequential([
                resize_and_rescale,
                layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
                layers.BatchNormalization(),  # Add Batch Normalization
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
                layers.BatchNormalization(),  # Add Batch Normalization
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
                layers.BatchNormalization(),  # Add Batch Normalization
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
                layers.BatchNormalization(),  # Add Batch Normalization
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
                layers.BatchNormalization(),  # Add Batch Normalization
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),  # Add Dropout
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),  # Add Dropout
                layers.Dense(num_classes, activation='softmax'),
            ])

            # Learning Rate Scheduler
            initial_learning_rate = 0.01
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            model.build(input_shape=input_shape)
            ModelBuilder.logger.info("Model built successfully.")
            ModelBuilder.logger.info(model.summary())
            return model
        except Exception as e:
            ModelBuilder.logger.error(f"An error occurred during model building: {e}")
            raise
