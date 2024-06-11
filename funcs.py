import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def count_labels(df):
    dic = {}
    unique_values, counts = np.unique(df, return_counts=True)
    for value, count in zip(unique_values, counts):
        dic[value] = count
    return dic


def count_white_pixels(df, labels):
    dic = dict.fromkeys(np.unique(labels), 0)
    for value, label in zip(df, labels):
        add = np.sum(value == 1)
        dic[label] += add
    return dic


def count_non_white_pixels(df, labels):
    dic = dict.fromkeys(np.unique(labels), 0)
    for value, label in zip(df, labels):
        add = np.sum(value != 1)
        dic[label] += add
    return dic


def calculate_statistics(df):
    average = {k: np.mean(v) for k, v in df.items()}
    std = {k: np.std(v) for k, v in df.items()}
    return average, std


def replace_pixel_by_average(df):
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9.0

    images_filtered = np.zeros_like(df)
    for i in range(df.shape[0]):
        images_filtered[i] = tf.nn.conv2d(df[i:i + 1, ..., np.newaxis], kernel[..., np.newaxis, np.newaxis],
                                          strides=[1, 1, 1, 1], padding='SAME').numpy().squeeze()
    return images_filtered


def print_scores(y_test, y_preds):
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average='weighted')
    recall = recall_score(y_test, y_preds, average='weighted')
    f1 = f1_score(y_test, y_preds, average='weighted')

    print()
    print("Accuracy score:", accuracy)
    print("Precision score:", precision)
    print("Recall (Sensitivity) score:", recall)
    print("F1 score:", f1)
    print()


def plot_conf_matrix(y_test, y_preds, keys, model_name):
    conf_matrix = confusion_matrix(y_test, y_preds)
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=keys)
    cm_disp.plot()
    plt.title(model_name)
    plt.show()


def plot_loss_to_epochs(histories, model_name):
    plt.figure(figsize=(14, 7))
    for epochs, history in histories.items():
        plt.plot(history.history['loss'], label=f'Training Loss (epochs={epochs})')
        plt.plot(history.history['val_loss'], label=f'Validation Loss (epochs={epochs})')

    plt.title(f'{model_name}: Epochs to Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def avaraging_img(df):
    sum_of_kernel = 0
    for y in range(0, 28, 3):
        for x in range(0, 28, 3):
            for ind in range(3):
                sum_of_kernel += df[x][y + ind]
                sum_of_kernel += df[x + ind][y]
                sum_of_kernel += df[x + ind][y + ind]
            average = sum_of_kernel / 9
            for ind in range(3):
                df[x][y + ind] = average
                df[x + ind][y] = average
                df[x + ind][y + ind] = average
    return df


def tp_count(y_test, y_preds):
    conf_matrix = confusion_matrix(y_test, y_preds)
    true_positives = np.diag(conf_matrix)
    return true_positives


def models_compare(y_test, y_preds1, y_preds2, y_preds3):
    tp1 = tp_count(y_test, y_preds1)
    tp2 = tp_count(y_test, y_preds2)
    tp3 = tp_count(y_test, y_preds3)

    x = [tp1, tp2, tp3]

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(x, color=["lightsalmon", "mediumaquamarine", "lightsteelblue"])
    plt.show()


def plot_img(df, model_name):
    sample = 1

    image = df[sample]

    fig = plt.figure
    plt.imshow(image, cmap='gray')
    plt.title(model_name)
    plt.show()


def block_averaging(df, block_size=3):
    pad_height = (block_size - df.shape[1] % block_size) % block_size
    pad_width = (block_size - df.shape[2] % block_size) % block_size
    padded_images = np.pad(df, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    new_height = padded_images.shape[1] // block_size
    new_width = padded_images.shape[2] // block_size
    reduced_images = np.zeros((df.shape[0], new_height, new_width))

    for i in range(df.shape[0]):
        for r in range(new_height):
            for c in range(new_width):
                block = padded_images[i, r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size]
                reduced_images[i, r, c] = np.mean(block)

    # Resize reduced images back to 28x28
    resized_images = np.zeros((df.shape[0], 28, 28))
    for i in range(df.shape[0]):
        resized_images[i] = tf.image.resize(reduced_images[i][np.newaxis, ..., np.newaxis], [28, 28]).numpy().squeeze()

    return resized_images


def undersample_images(df, new_size):
    # Convert the normalized image back to pixel values (0-255)
    new_df = np.zeros_like(a=df, shape=(df.shape[0], new_size[0], new_size[1]))
    for i in range(df.shape[0]):
        image = (df[i] * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        undersampled_image = pil_image.resize(new_size, Image.LANCZOS)
        new_df[i] = np.array(undersampled_image) / 255.0  # Normalize back to 0-1

    resized_images = np.zeros((df.shape[0], 28, 28))
    for i in range(df.shape[0]):
        resized_images[i] = tf.image.resize(new_df[i][np.newaxis, ..., np.newaxis], [28, 28]).numpy().squeeze()

    return resized_images


def rotate_images(df, angle):
    new_df = np.zeros_like(df)
    for i in range(df.shape[0]):
        image = df[i]
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_df[i] = cv2.warpAffine(image, rotation_matrix, (w, h))

    return new_df


def plot_rotated_image(image, rotated_image):
    # Display the original and rotated images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Rotated Image')
    plt.imshow(rotated_image, cmap='gray')

    plt.show()
