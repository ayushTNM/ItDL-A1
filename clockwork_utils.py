import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback, CSVLogger

import tensorflow as tf
import os
import time
from collections import deque
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
import csv


def create_log_dir(best_model_dir, marker_name):

    log_dir = './logs/log' + marker_name

    # Create a directory path for best model
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    return log_dir

    
def latest_checkpoint(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_")]
    latest = max(checkpoints, key=os.path.getctime)
    return latest

def get_callbacks(log_dir, checkpoint_path, patience, mode='classify'):

    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=patience)
    
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True)
    
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1)

    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss',         # Monitor validation loss, should stay in tune with training loss
        factor=0.6,                 # Factor by which to reduce the learning rate 
        patience=4,                 # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1.0e-9               # Lower bound on the learning rate
    )

    csv_logger = CSVLogger(f'{log_dir}_log.csv', append=True, separator=';')

    callbacks_list = [early_stopping, model_checkpoint, tensorboard_callback, csv_logger, lr_schedule]
    
    return callbacks_list

# Custom transformation function
def transform_labels(image, labels, mode, n_categories):
    hours, minutes = labels[0], labels[1]
    if mode == 'classify':
        minutes_per_class = int(720 / n_categories)
        number_minute_classes = int(60 / minutes_per_class)
        hour_classes = hours 
        minute_classes = minutes // minutes_per_class
        combined_labels = hour_classes * number_minute_classes + minute_classes

    elif mode == 'regress':
        hours_float = tf.cast(hours, tf.float32)
        minutes_float = tf.cast(minutes, tf.float32)
        combined_labels = (hours_float * 60 + minutes_float) / 720.0

    elif mode == 'multihead':
        minutes_per_class = int(720 / n_categories)
        number_minute_classes = int(60 / minutes_per_class)
        hour_classes = hours 
        minute_classes = minutes // minutes_per_class
        combined_labels = (hours, minute_classes)
    
    elif mode == 'cyclic':
        hours_angle = tf.cast(hours, tf.float32) * (2 * np.pi / 12)  
        minutes_angle = tf.cast(minutes, tf.float32) * (2 * np.pi / 60) 
        sin_hours = tf.sin(hours_angle)
        cos_hours = tf.cos(hours_angle)
        sin_minutes = tf.sin(minutes_angle)
        cos_minutes = tf.cos(minutes_angle)
        combined_labels = [sin_hours, cos_hours, sin_minutes, cos_minutes]

    elif mode == 'multicyclic':
        hour_classes = hours 
        minutes_angle = tf.cast(minutes, tf.float32) * (2 * np.pi / 60) 
        sin_minutes = tf.sin(minutes_angle)
        cos_minutes = tf.cos(minutes_angle)
        combined_labels = (hours, sin_minutes, cos_minutes)
        
    else:
        raise ValueError('invalid type')
    
    return image, combined_labels

# Normalize the images
def normalize_img(img, label):
    return tf.cast(img, tf.float32) / 255., label

def load_and_process_data(data_dir, num_train, num_val, batch_size, mode, n_categories, random_seed):
    # Load data, we will keep it in numpy
    labels = np.load(data_dir + 'labels.npy')
    # labels = labels[..., np.newaxis]  # Add an extra dimension for channels
    images = np.load(data_dir + 'images.npy')
    images = images[..., np.newaxis]  # Add an extra dimension for channels

    # Create a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Shuffle the dataset
    buffer_size = len(images)
    dataset = dataset.shuffle(buffer_size=buffer_size, seed=random_seed)
    dataset = dataset.map(lambda image, label: normalize_img(image, label))
    dataset = dataset.map(lambda image, label: transform_labels(image, label, mode, n_categories))

    train_dataset = dataset.take(num_train).batch(batch_size)
    remain_dataset = dataset.skip(num_train)
    val_dataset = remain_dataset.take(num_val).batch(batch_size)
    test_dataset = remain_dataset.skip(num_val).batch(batch_size)
    
    return train_dataset, val_dataset, test_dataset

def get_compiler(mode):
    # [Model compilation, summary] 
    if mode == 'classify':
        loss={'t': 'sparse_categorical_crossentropy'}
        loss_weights={'t': 1},        
        metrics={'t': 'accuracy'}  

    elif mode == 'regress':
        loss={'t': 'mae'}
        loss_weights={'t': 1},        
        metrics={'t': 'mse'}

    elif mode == 'multihead':
        loss={'h': 'sparse_categorical_crossentropy', 'm': 'sparse_categorical_crossentropy'}
        loss_weights={'h': 1., 'm': 1.},        
        metrics={'h': 'accuracy', 'm': 'accuracy'}  

    elif mode == 'cyclic':
        loss={'sh': 'mae', 'ch': 'mae', 'sm': 'mae', 'cm': 'mae'}
        loss_weights={'sh': 1., 'ch': 1., 'sm': 1., 'cm': 1.},        
        metrics={'sh': 'mse', 'ch': 'mse', 'sm': 'mse', 'cm': 'mse'}  

    elif mode == 'multicyclic':
        loss={'h': 'sparse_categorical_crossentropy', 'sm': 'mae', 'cm': 'mae'}
        loss_weights={'h': 1., 'sm': 1., 'cm': 1.},        
        metrics={'h': 'accuracy', 'sm': 'mse', 'cm': 'mse'} 
        
    return loss, metrics, loss_weights

# def append_to_file(file_path, text_to_append):
#     # 'a' mode opens the file for appending
#     with open(file_path, 'a') as file:
#         file.write(text_to_append + '\n')  
#         # '\n' will move to the next line after each entry

# def write_to_csv(file_path, data):
#     # The 'a' parameter allows you to append to the end of the file when writing
#     with open(file_path, 'a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(data)

# def log_results(file_path, header, results):
#     # Check if log file exists to decide whether to write headers
#     file_exists = os.path.isfile(file_path)
    
#     with open(file_path, 'a', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=header)
        
#         if not file_exists:
#             writer.writeheader()  # File didn't exist, we had to create a new one, write the header

#         writer.writerow(results)

# def read_from_csv(file_path):
#     with open(file_path, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         return list(reader)


# def convert_sin_cos_to_time(sin_cos_values):

#     # Ensure the input is numpy array
#     sin_cos_values = np.array(sin_cos_values)
    
#     # Calculate the angle in radians for hours and minutes from the sine and cosine values
#     hours_angle = np.arctan2(sin_cos_values[:, 0], sin_cos_values[:, 1])  # atan2(sin, cos)
#     minutes_angle = np.arctan2(sin_cos_values[:, 2], sin_cos_values[:, 3])

#     # Convert radians to degrees to get the hour (we assume 12 hours in a circle, not 24)
#     hours = np.mod((hours_angle * (12 / (2 * np.pi))), 12)  # Convert radian to hour
#     minutes = np.mod((minutes_angle * (60 / (2 * np.pi))), 60)  # Convert radian to minute

#     return hours * 60 + minutes


# def mean_circular_error(y_true, y_pred):

#     # Convert minutes to radians, where 12 hours (720 minutes) is the full circle (2*pi)
#     y_true_rad = np.radians((y_true / 720) * 360)
#     y_pred_rad = np.radians((y_pred / 720) * 360)

#     # Calculate the differences considering the circular nature
#     angular_difference = np.arctan2(np.sin(y_true_rad - y_pred_rad), np.cos(y_true_rad - y_pred_rad))
    
#     # Convert differences back to minutes
#     error_minutes = np.abs(np.degrees(angular_difference) / 360 * 1440)
    
#     # Return the mean error
#     return np.mean(error_minutes)

# def plot_accuracy_curves(file_path):
#     # Read data from CSV
#     csv_data = read_from_csv(file_path)

#     # Extract header and data
#     header, *data_rows = csv_data

#     # Separate data based on type (assumes 'type' is the second column)
#     val_data = [row for row in data_rows if row[1] == 'val_acc']
#     train_data = [row for row in data_rows if row[1] == 'train_acc']

#     # Prepare your figure with subplots
#     fig, axs = plt.subplots(1, 2, figsize=(15, 5))  # 1x3 subplots

#     # Set a log scale for y-axis
#     for ax in axs:
#         ax.set_yscale('log')
#         ax.set_ylim(0.01, 1)  # Set the limits to be between 0.01 and 1
#         ax.grid(True, which="both", ls="--", linewidth=0.5)
#         ax.minorticks_on()  # Ensure minor ticks are on

#     # Function to process and plot data
#     def plot_data(data, ax, label_prefix):
#         for index, row in enumerate(data):
#             run_number = row[0]  # assumes 'index' is the first column
#             accuracies = [float(val) for val in row[8:] if val]  # Convert to float

#             # Plot the accuracy curve
#             ax.plot(range(1, len(accuracies) + 1), accuracies, label=f"{label_prefix} {run_number}")

#     # Plot the data
#     plot_data(val_data, axs[0], "Val Run")
#     plot_data(train_data, axs[1], "Train Run")

#     # Set titles and labels
#     axs[0].set_title('Validation Accuracy')
#     axs[1].set_title('Training Accuracy')
#     for ax in axs:
#         ax.set_xlabel('Epochs')
#         ax.set_ylabel('Accuracy')
#         ax.legend(loc='best', fontsize='small')

#     plt.tight_layout()
#     plt.show()


def extract_accuracy_prediction(test_dataset, model):
    # Extracting images and their true labels
    test_images = []
    test_labels = []
    for images, labels in test_dataset:
        test_images.append(images.numpy())
        test_labels.append(labels.numpy())

    # Converting list to numpy array for computation
    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)

    # Predicting using the model
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)  # Getting the index of the class with the highest probability

    return test_labels, predicted_labels, test_images


# def plot_accuracy_history(history):
#     xmax = len(history.history['accuracy'])

#     # summarize history for accuracy
#     plt.figure(figsize=(10, 5))
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.yscale('log')
#     plt.ylim([0.01,1])
#     plt.xlim([0,xmax])
#     plt.grid(which='major', linestyle='-', linewidth='1')
#     plt.grid(which='minor', linestyle='--',linewidth='0.5')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()

#     # summarize history for loss
#     plt.figure(figsize=(10, 5))
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.yscale('log')
#     plt.ylim([1,10])
#     plt.xlim([0,xmax])
#     plt.grid(which='major', linestyle='-', linewidth='1')
#     plt.grid(which='minor', linestyle='--',linewidth='0.5')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
#     plt.show()

#     return


# # # def plot_accuracy_predictions(model, dataset, class_names, n_categories, n_batches=113, marker_name=""):
# # #     m = int(720 / n_categories)
# # #     n = int(60 / m)
# # #     max_plots = 64  # Max number of plots for an 8x8 grid

# # #     # First, gather all predictions and labels for the given batches
# # #     all_images = []
# # #     all_labels = []
# # #     all_predictions = []
# # #     for images, labels in dataset.take(n_batches):  # Only take n_batches
# # #         all_images.extend(images.numpy())
# # #         all_labels.extend(labels.numpy())
# # #         preds = model.predict(images)
# # #         all_predictions.extend(np.argmax(preds, axis=1))

# # #     # Now filter out correct predictions
# # #     incorrect_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions)) if true != pred]

# # #     plt.figure(figsize=(10, 10))
# # #     plot_idx = 0  # This will also serve as a plot counter to avoid exceeding 64 plots
# # #     for index in incorrect_indices:
# # #         if plot_idx >= max_plots:
# # #             break  # Avoid plotting more than 64 images
        
# # #         images_np = all_images[index]
# # #         true_label_int = all_labels[index]
# # #         predicted_label_int = all_predictions[index]
# # #         true_h = int(true_label_int / n)
# # #         true_m = int(true_label_int % n) * m
# # #         pred_h = int(predicted_label_int / n)
# # #         pred_m = int(predicted_label_int % n) * m

# # #         class_error = int((true_label_int - predicted_label_int + n_categories) % n)
# # #         class_error = min(class_error, n - class_error)

# # #         color = "orange" if int(class_error) <= m else "red"
# # #         plot_idx += 1
# # #         plt.subplot(8, 8, plot_idx)
# # #         plt.xticks([])
# # #         plt.yticks([])
# # #         plt.grid(False)
# # #         plt.imshow(images_np.squeeze(), cmap='gray')  # Assuming grayscale
# # #         plt.title(f"True: {true_h:02d}:{true_m:02d}\nPred: {pred_h:02d}:{pred_m:02d}", color=color, fontsize=10)

# # #     plt.tight_layout()
# # #     plt.savefig(f"accuracy_{marker_name}.pdf")  # Ensure marker_name is a valid string
# # #     plt.show()
# # #     plt.clf()  # Clear figure to free up memory

# # #     return all_predictions  # Or you could return just the incorrect ones if needed


# # # def plot_confusion_matrix(test_labels, predicted_labels, n_categories):

# # #     # Assuming you have 'num_classes' classes in your classification task
# # #     cm = confusion_matrix(test_labels, predicted_labels, labels=range(n_categories))
# # #     plt.figure(figsize=(10, 8))
# # #     heatmap = sns.heatmap(cm, annot=False, fmt="d", cmap='Blues', 
# # #                         xticklabels=range(n_categories), 
# # #                         yticklabels=range(n_categories), 
# # #                         annot_kws={"size":3})
# # #     plt.title('Confusion matrix')
# # #     plt.ylabel('Actual label')
# # #     plt.xlabel('Predicted label')
# # #     plt.show()

# # #     return

# # def calculate_hour_accuracy(test_labels, predicted_labels, segments_per_hour=6):
# #     # Convert category labels back to hour format (integer division)
# #     test_hours = np.array(test_labels) // segments_per_hour
# #     predicted_hours = np.array(predicted_labels) // segments_per_hour

# #     # Calculate accuracy for hours specifically
# #     hour_accuracy = accuracy_score(test_hours, predicted_hours)

# #     return hour_accuracy

# def print_score_summary(test_labels, predicted_labels):

#     accuracy = accuracy_score(test_labels, predicted_labels)
#     f1 = f1_score(test_labels, predicted_labels, average='macro')  # Choose appropriate averaging for your case
#     recall = recall_score(test_labels, predicted_labels, average='macro')  # Same as above

#     print(f"Accuracy: {accuracy:6.4f}")
#     print(f"F1 Score: {f1:6.4f}")
#     print(f"Recall: {recall:6.4f}")

#     return