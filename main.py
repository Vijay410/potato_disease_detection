import os
import logging
from logging_config import setup_custom_logger
from preprocessing import ImageDataPipeline
from model import ModelBuilder
from train import Trainer
from evaluate import model_evaluate, plot_training_history
from predict import predict_sample_image, predict, inference

# Set up logger
logger = setup_custom_logger(__name__)
#logger
if __name__ == "__main__":
    try:
        data_dir = '/home/stellapps/potato_disease_detection/dataset'
        result = '/home/stellapps/potato_disease_detection/results'
        IMAGE_SIZE = 256
        BATCH_SIZE = 32
        epochs = 100
        CHANNELS = 3
        num_classes = 3  # Adjust this according to your dataset
        input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

        # Example usage:
        pipeline = ImageDataPipeline(data_dir)
        dataset = pipeline.load_dataset()
        class_names = dataset.class_names
        train_ds, val_ds, test_ds = pipeline.partition_dataset(dataset)
        train_ds = pipeline.cache_shuffle_prefetch(train_ds)
        val_ds = pipeline.cache_shuffle_prefetch(val_ds)
        test_ds = pipeline.cache_shuffle_prefetch(test_ds)
        resize_and_rescale = pipeline.resize_and_rescale_model()
        data_augmentation = pipeline.data_augmentation_model()
        train_ds = pipeline.apply_data_augmentation(train_ds, data_augmentation)
        model = ModelBuilder.build_model(input_shape, num_classes, resize_and_rescale)
        history = Trainer.train_model(model, train_ds, val_ds, epochs, BATCH_SIZE)
        scores, acc, loss, val_loss, val_acc = model_evaluate(model, test_ds, history)
        plot_training_history(acc, val_acc, loss, val_loss, epochs, result)
        predict_sample_image(test_ds, class_names, model, result)
        inference(model, test_ds, class_names)

        # Define the full path including filename and extension
        model_directory = "/home/stellapps/potato_disease_detection/models/"
        model_filename = "leafe_detection_model_v1.h5"
        model_path = os.path.join(model_directory, model_filename)

        # Ensure the directory exists, if not, create it
        os.makedirs(model_directory, exist_ok=True)

        # Save the model
        model.save(model_path)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
