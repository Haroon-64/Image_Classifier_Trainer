# visualizer.py
import matplotlib.pyplot as plt

def show_metrics(model):
    """
    return the training and validation metrics as plot
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(model.history['accuracy'])
    ax[0].plot(model.history['val_accuracy'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(['Train', 'Validation'], loc='upper left')

    ax[1].plot(model.history['loss'])
    ax[1].plot(model.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend(['Train', 'Validation'], loc='upper left')
    return fig

def show_predictions(model, images, labels):
    predictions = model.predict(images)
    predicted_classes = predictions.argmax(axis=-1)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        ax[i].imshow(images[i])
        ax[i].set_title(f'Actual: {labels[i]}, Predicted: {predicted_classes[i]}')
        ax[i].axis('off')
    return fig

def Gcam(model, image, layer_name):
    """
    return the Grad-CAM heatmap for the given image and layer
    """
    pass

