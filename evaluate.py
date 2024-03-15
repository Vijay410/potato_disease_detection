import json
import os
import matplotlib.pyplot as plt

def model_evaluate(model, test_ds, history):
    scores = model.evaluate(test_ds)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    evaluation_result = {
        'Accuracy': acc,
        'validation_accuracy': val_acc,
        'loss': loss,
        'validation_loss': val_loss,
        'scores': scores
    }
    file_path = os.path.join('/home/stellapps/potato_disease_detection/results', 'evaluation_scores.json')
    with open(file_path, 'w') as json_file:
        json.dump(evaluation_result, json_file)

    return scores, acc, loss, val_loss, val_acc

def plot_training_history(acc, val_acc, loss, val_loss, EPOCHS, folder_path):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label='Training Accuracy', color='blue')
    plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy', color='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss, label='Training Loss', color='blue')
    plt.plot(range(EPOCHS), val_loss, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    # Save the plot as an image file in the specified folder
    file_path = os.path.join(folder_path, 'training_history_plot.png')
    plt.savefig(file_path)
