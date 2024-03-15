import tensorflow as tf
from logging_config import setup_custom_logger

class Trainer:
    logger = setup_custom_logger(__name__)

    @staticmethod
    def train_model(model, train_ds, val_ds, epochs, batch_size):
        try:
            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy']
            )
            history = model.fit(
                train_ds,
                batch_size=batch_size,
                validation_data=val_ds,
                verbose=1,
                epochs=100,
            )

            Trainer.logger.info("Model training completed successfully.")
            return history
        except Exception as e:
            Trainer.logger.error(f"An error occurred during model training: {e}")
            raise
