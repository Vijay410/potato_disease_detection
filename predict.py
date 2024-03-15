import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from logging_config import setup_custom_logger

logger = setup_custom_logger(__name__)

def predict_sample_image(test_ds, class_names, model, result):
    try:
        for images_batch, labels_batch in test_ds.take(1):
            first_image = images_batch[0].numpy().astype('uint8')
            first_label = labels_batch[0].numpy()
            
            logger.info("Displaying first image to predict...")
            plt.imshow(first_image)
            logger.info(f"Actual label: {class_names[first_label]}")

            batch_prediction = model.predict(images_batch)
            predicted_label_index = np.argmax(batch_prediction[0])
            predicted_label = class_names[predicted_label_index]

            logger.info(f"Predicted label: {class_names[np.argmax(batch_prediction[0])]}")

            image_filename = os.path.join(result, f"predicted_image_{predicted_label}.jpg")
            plt.imsave(image_filename, first_image)
            logger.info(f"Predicted image saved as: {image_filename}")

    except Exception as e:
        logger.error(f"An error occurred during predicting sample image: {e}")
        raise

def predict(model, img, class_names):
    try:
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)

        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * (np.max(predictions[0])), 2)
        return predicted_class, confidence

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        raise

def inference(model, test_ds, class_names):
    try:
        plt.figure(figsize=(15, 15))
        for images, labels in test_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                
                predicted_class, confidence = predict(model, images[i].numpy(), class_names)
                actual_class = class_names[labels[i]] 
                
                plt.title(f"Actual: {actual_class}, Predicted: {predicted_class}. Confidence: {confidence}%")
                plt.axis("off")
                image_filename = os.path.join('/home/stellapps/potato_disease_detection/results/predicted_samples', f"image_{i+1}_predicted_as_{predicted_class}.jpg")
                plt.savefig(image_filename, bbox_inches='tight', pad_inches=0)
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        raise
