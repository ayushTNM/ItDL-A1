#!/usr/bin/env python
# coding: utf-8

# PROVIDE DIRECTORY OF CLOCK IMAGES AND LABELS HERE (DATA NOT INCLUDED IN SUBMISSION)
data_dir = './data/CLOCKS/'

# 0: GENERIC TRAINING METHOD AND CNN MODEL (GENERATES TABLE 1) CAN BE USED TO GENERATE ANY OF THE MODELS
# 1: LOAD BEST CLASSIFICATION MODEL AND SHOW MISCLASSIFIED IMAGES (FIGURE 1)
# 2: STATISTICS CLASSIFICATION (TABLE 2)
# 3: LOAD BEST REGRESSION MODEL
# 4: SHOW RESULTS REGRESSION (FIGURE 2)
# 5: ANALYSE MULTIHEAD PREDICTIONS (FIGURE 3)
# 6: LOAD BEST MULTICYCLIC MODEL
# 7: ANALYSE BEST MULTICYCLIC MODEL (FIGURE 4)

load_existing = True

# 0: GENERIC TRAINING METHOD (TABLE 1) >> generate all models from here
import os
import csv
import keras
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import clockwork_model as cw
import clockwork_utils as cu
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

tf.config.set_visible_devices([], 'GPU')

if not load_existing:
    def main(num_kernels, learning_rate, depth, dropout, mode):

        random_seed = 42

        # data parameters
        num_images = 18000
        train_split = 0.64   # 80% x 80%
        val_split = 0.16     # 80% x 20%
        test_split = 1 - train_split - val_split

        #architecture
        input_shape = (150, 150, 1)
        n_categories = int (720 / 10)
        num_kernels=num_kernels
        depth=depth
        dropout1=dropout
        dropout2=dropout/2
        learning_rate = learning_rate

        # hyper parameters
        mode = mode
        batch_size = 32
        epochs = 50
        patience = 6

        # locations of i/o
        data_dir = './data/CLOCKS/'
        marker_name = f'{mode}_{num_kernels}_{depth}_{learning_rate:.2e}_{dropout1:.2e}:{batch_size}_{random_seed}'
        best_model_dir = './best_model'
        best_model_path = best_model_dir + f'/best_model_{marker_name}'
        checkpoint_dir = f'./training/training_{marker_name}'

        log_dir, checkpoint_path = cu.create_check_log_dir(checkpoint_dir, marker_name)

        # calculate sizes of splits for later
        num_train = int(train_split * num_images)
        num_val = int(val_split * num_images)
        num_test = num_images - num_train - num_val
        # class_names = [str(i) for i in range(n_categories)]

        print(f"loading from {data_dir}")
        train_dataset, val_dataset, test_dataset = cu.load_and_process_data(data_dir, num_train, num_val, batch_size, mode)

        # [Model loading or creation]
        model = cw.create_model(
            mode=mode,
            input_shape=input_shape,
            n_categories=n_categories,
            num_kernels=num_kernels,
            depth=depth,
            dropout1=dropout1,
            dropout2=dropout2)

        compiler_loss, compiler_metrics = cu.get_compiler(mode)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=compiler_loss,
            metrics=compiler_metrics)

        model.summary()

        callbacks_list = cu.get_callbacks(model, train_dataset, log_dir, checkpoint_path, patience, mode)

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,  # Use the validation set here
            steps_per_epoch=num_train // batch_size,
            validation_steps=num_val // batch_size,  # steps for the validation set
            verbose=1,
            epochs=epochs,
            callbacks=callbacks_list)

        # Load latest checkpoint file to best model location
        best_model_file = cu.latest_checkpoint(checkpoint_dir)
        best_model = load_model(best_model_file)
        best_model.save(best_model_path)


    if __name__ == "__main__":

        # settings to generate best classification model for 720 classes
        mode_list=['classify', 'regress', 'multihead', 'cyclic']
        mode = mode_list[0]
        num_kernels_list=[(32, 64, 128, 256, 256)]
        depth_list=[None]
        lr_list = [1.e-4]
        dropout_list = [0.02]
        n_categories = [720]

        log_file_path = f'./best_model/model_{mode}_logs.csv'
        header = ['index', 'nk_idx', 'd_idx', 'lr_idx', 'do_idx'] + [f'epoch_{i}' for i in range(1, 51)]  # max epochs = 50
        with open(log_file_path, 'w', newline='') as file:  # Opening the file in write mode to create it/reset it
            writer = csv.writer(file)
            writer.writerow(header)  # Writing the header

        for nk_idx, num_kernels in enumerate(num_kernels_list):
            for d_idx, depth in enumerate(depth_list):
                for lr_idx, lr in enumerate(lr_list):
                        for do_idx, dropout in enumerate(dropout_list):
                            main(num_kernels, lr, depth, dropout, mode)
                            print(f"{mode} on {n_categories} classes : kernel = {num_kernels} depth = {depth}, learning rate = {lr}, dropout = {dropout}")

        # Store the data here
        rows = []
        with open(log_file_path, 'r') as csvfile:
            # Create a CSV reader
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                rows.append(row)


# 1: LOAD BEST CLASSIFICATION MODEL AND SHOW MISCLASSIFIED IMAGES (FIGURE 1)

# EXISTING DATA:
# ./best_model/best_model_classify_720_(32, 64, 128, 256, 256)_None_1.00e-04_2.00e-02:32_42
# ./best_model/best_model_classify_360_(32, 64, 128, 256, 256)_None_1.00e-04_2.00e-02:32_42
# ./best_model/best_model_classify_240_(32, 64, 128, 256, 256)_None_1.00e-04_2.00e-02:32_42
# ./best_model/best_model_classify_180_(32, 64, 128, 256, 256)_None_1.00e-04_2.00e-02:32_42
# ./best_model/best_model_classify_144_(32, 64, 128, 256, 256)_None_1.00e-04_2.00e-02:32_42
# ./best_model/best_model_classify_120_(32, 64, 128, 256, 256)_None_1.00e-04_2.00e-02:32_42

random_seed = 42

# data parameters
mode='classify'
num_images = 18000
train_split = 0.64   # 80% x 80%
val_split = 0.16     # 80% x 20%

# Choices go here
# number of classes for time-telling
n_categories_list = [120, 144, 180, 240, 360, 720]
# different models and labels, can run in batch or just select one model
mode_list=['classify', 'regress', 'multihead', 'cyclic', 'multicyclic']
# kernels for the 5 convolutional model
num_kernels_list=[(32, 64, 128, 256, 256), (32, 32, 64, 64, 64)]
# use of depth kernel
depth_list=['None', 3, 4]
# initial learning rate
lr_list = [1.e-4, 2.e-4, 5.e-4]
# intial drop-out
dropout_list = [0., 1.e-2, 1.3e-2, 2.e-2, 1.e-1, 2.e-1, 5.e-1]

# best model for classification goes here
mode = mode_list[0]
n_categories = n_categories_list[5]
num_kernels = num_kernels_list[0]
depth = depth_list[0]
learning_rate = lr_list[0]
drop_out = dropout_list[3]

# architecture
input_shape = (150, 150, 1)
batch_size = 32

# index to keep track if we run multiple models simultaneously
global_index = 0

# locations of data
data_dir = './data/CLOCKS/'

# run generic loader from here
# calculate sizes of splits for later
num_train = int(train_split * num_images)
num_val = int(val_split * num_images)
num_test = num_images - num_train - num_val

class_names = [str(i) for i in range(n_categories)]

train_dataset, val_dataset, test_dataset = cu.load_and_process_data(data_dir, num_train, num_val, batch_size, mode, n_categories, random_seed)

best_model_file = f'./best_model/best_model_{mode}_{n_categories}_{num_kernels}_{depth}_{learning_rate:.2e}_{drop_out:.2e}:{batch_size}_{random_seed}'
print(best_model_file)

best_model = load_model(best_model_file)

test_labels, predicted_labels, test_images = cu.extract_accuracy_prediction(test_dataset, best_model)

# Calculate metrics
accuracy = accuracy_score(test_labels, predicted_labels)
recall = recall_score(test_labels, predicted_labels, average='macro')  # 'macro' for unweighted mean
precision = precision_score(test_labels, predicted_labels, average='macro')
f1 = f1_score(test_labels, predicted_labels, average='macro')

incorrect_labels = np.where(predicted_labels != test_labels)[0]

# Calculate the error distances for incorrect predictions
error_distances = (predicted_labels[incorrect_labels] - test_labels[incorrect_labels])
error_distances = np.abs(error_distances) % n_categories  # Ensure error wraps around for circular categories

# Correct for wrap-around at the halfway point
halfway_point = n_categories // 2
error_distances = np.where(error_distances > halfway_point, n_categories - error_distances, error_distances)

# Count the frequency of each error distance
error_distance_counts = np.bincount(error_distances)

# For presentation, let's create a dictionary that skips the 0 error (since we're only interested in incorrect ones)
error_summary = {distance: count for distance, count in enumerate(error_distance_counts) if distance > 0 and count > 0}

# Calculate parameters for time conversion
m = int(720 / n_categories)
n = int(60 / m)

# Set up the plot
num_incorrect = len(incorrect_labels)
if num_incorrect > 0:
    columns = 8
    rows = (num_incorrect + columns - 1) // columns  # Ensure enough rows to show all incorrect images
    plt.figure(figsize=(2 * columns, 2 * rows))

    # Loop through the incorrect predictions
    for plot_idx, index in enumerate(incorrect_labels):
        image = test_images[index]
        true_label_int = test_labels[index]
        predicted_label_int = predicted_labels[index]

        true_h = int(true_label_int / n)
        true_m = int(true_label_int % n) * m
        pred_h = int(predicted_label_int / n)
        pred_m = int(predicted_label_int % n) * m

        class_error = int((true_label_int - predicted_label_int + n_categories) % n_categories)
        class_error = min(class_error, n_categories - class_error)

        # Determine color based on the magnitude of the error
        if class_error == 0:
            color = 'green'
        elif class_error <= m:
            color = 'orange'
        else:
            color = 'red'

        plt.subplot(rows, columns, plot_idx + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {true_h:02d}:{true_m:02d}\nPred: {pred_h:02d}:{pred_m:02d}", color=color, fontsize=12)

    plt.tight_layout()
    plt.savefig("misclassified test.pdf")
    plt.show()


# 2: STATISTICS CLASSIFICATION (TABLE 2)

test_hour_labels = np.floor_divide(test_labels, 60)
test_min_labels = np.mod(test_labels, 60)
pred_hour_labels = np.floor_divide(predicted_labels, 60)
pred_min_labels = np.mod(predicted_labels, 60)
accuracy_hour = accuracy_score(test_hour_labels, pred_hour_labels)
recall_hour = recall_score(test_hour_labels, pred_hour_labels, average='macro')  # 'macro' for unweighted mean
precision_hour = precision_score(test_hour_labels, pred_hour_labels, average='macro')
f1_hour = f1_score(test_hour_labels, pred_hour_labels, average='macro')
accuracy_min = accuracy_score(test_min_labels, pred_min_labels)
recall_min = recall_score(test_min_labels, pred_min_labels, average='macro')  # 'macro' for unweighted mean
precision_min = precision_score(test_min_labels, pred_min_labels, average='macro')
f1_min = f1_score(test_min_labels, pred_min_labels, average='macro')
print("Latex output for report")
print(f'{int(720/n_categories)} & {n_categories} & {100* accuracy_hour:6.2f}\% & {100*precision_hour:6.2f}\% & {100*recall_hour:6.2f}\% & {100*f1_hour:6.2f}\% \\\\')
print(f'{int(720/n_categories)} & {n_categories} & {100* accuracy_min:6.2f}\% & {100*precision_min:6.2f}\% & {100*recall_min:6.2f}\% & {100*f1_min:6.2f}\% \\\\')


# 3: LOAD BEST REGRESSION MODEL

# EXISTING DATA:
# ./best_model/best_model_regress_720_(32, 64, 128, 256, 256)_None_1.00e-04_1.30e-02:32_42


if load_existing:

    # Choices go here
    # number of classes for time-telling
    n_categories_list = [120, 144, 180, 240, 360, 720]
    # different models and labels, can run in batch or just select one model
    mode_list=['classify', 'regress', 'multihead', 'cyclic', 'multicyclic']
    # kernels for the 5 convolutional model
    num_kernels_list=[(32, 64, 128, 256, 256), (32, 32, 64, 64, 64)]
    # use of depth kernel
    depth_list=['None', 3, 4]
    # initial learning rate
    lr_list = [1.e-4, 2.e-4, 5.e-4]
    # intial drop-out
    dropout_list = [0., 1.e-2, 1.3e-2, 2.e-2, 1.e-1, 2.e-1, 5.e-1]

    # best model for classification goes here
    mode = mode_list[1]
    n_categories = n_categories_list[4]
    num_kernels = num_kernels_list[0]
    depth = depth_list[0]
    learning_rate = lr_list[0]
    drop_out = dropout_list[2]

    random_seed = 42
    mode = 'regress'
    num_images = 18000
    train_split = 0.64   # 80% x 80%
    val_split = 0.16     # 80% x 20%

    #architecture
    input_shape = (150, 150, 1)
    n_categories = 720
    batch_size = 32
    class_names = [str(i) for i in range(n_categories)]

    # locations of i/o
    data_dir = './data/CLOCKS/'
    best_model_file = f'./best_model/best_model_{mode}_{n_categories}_{num_kernels}_{depth}_{learning_rate:.2e}_{drop_out:.2e}:{batch_size}_{random_seed}'
    print(best_model_file)

    # calculate sizes of splits for later
    num_train = int(train_split * num_images)
    num_val = int(val_split * num_images)
    num_test = num_images - num_train - num_val

    train_dataset, val_dataset, test_dataset = cu.load_and_process_data(data_dir, num_train, num_val, batch_size, mode, n_categories, random_seed)
    best_model = load_model(best_model_file)


# 4: SHOW RESULTS REGRESSION (FIGURE 2)

def calculate_r_squared(actual, predicted):
    # Calculate the mean of the actual values
    mean_actual = np.mean(actual)

    # Calculate total sum of squares
    total_sum_of_squares = np.sum((actual - mean_actual) ** 2)

    # Calculate the residual sum of squares
    residual_sum_of_squares = np.sum((actual - predicted) ** 2)

    # Calculate the R² score
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

    return r_squared

def calculate_residual_statistics(actual, predicted):
    residuals = actual - predicted
    std_residuals = np.std(residuals)
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    return residuals, std_residuals, mae, mse, rmse

def plot_regression_results(model, dataset, dataset_name=""):

    # First, concatenate the batches in the dataset into one large batch
    images = []
    labels = []
    for image_batch, label_batch in dataset:
        images.append(image_batch)
        labels.append(label_batch)

    # Stack the batches (now the entire dataset is in memory)
    images = tf.concat(images, axis=0)
    labels = tf.concat(labels, axis=0)

    # Get the actual and predicted values
    predictions = 720 * model.predict(images).squeeze()
    actual_values = 720 * labels.numpy()  # Convert to numpy array

    r_squared = calculate_r_squared(actual_values, predictions)
    residuals, std_residuals, mae, mse, rmse = calculate_residual_statistics(actual_values, predictions)

    print("regression R² score:", r_squared)
    print(f"Standard Deviation of Residuals: {std_residuals}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    # Assuming 'predictions' are the continuous outputs from your regression model
    predicted_classes = np.rint(720 * predictions).astype(int)
    print(predicted_classes)

    # Start a new figure
    plt.figure(figsize=(14, 6))

    # Create the first subplot for the regression scatter plot
    plt.subplot(1, 2, 1)  # (rows, columns, panel number)
    plt.scatter(actual_values, predictions, alpha=0.6)  # alpha for transparency

    # Add a line for perfect correlation. This is where actual == predicted
    max_val = max(actual_values.max(), predictions.max())
    min_val = min(actual_values.min(), predictions.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)  # 'k--' is style for black dashed line
    plt.text(min_val+50, max_val-50, f'R² = {r_squared:.2f}', fontsize=14, fontweight='bold', verticalalignment='top', horizontalalignment='left')

    # Add titles and labels
    plt.title(f'{dataset_name} Data: Actual vs. Predicted', fontsize=14, fontweight='bold')
    plt.xlim([0,720])
    plt.ylim([0,720])
    plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.grid(True)

    # Show the plot
    # Create the second subplot for the histogram of residuals
    plt.subplot(1, 2, 2)
    bin_width = 2  # bin width set to 2 minutes
    min_residual = np.floor(min(residuals))
    max_residual = np.ceil(max(residuals))

    # Create bins from min to max with a step of 2 minutes
    bins = np.arange(min_residual, max_residual + bin_width, bin_width)
    count, bins, ignored = plt.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black', zorder=3)
    max_count = 50 * (1 + int(max(count)/50))
    # Draw fill_between behind the histogram by setting a lower zorder
    plt.fill_betweenx([0, max_count], -2*std_residuals, 2*std_residuals, color='green', alpha=0.2, zorder=1)
    plt.fill_betweenx([0, max_count], -std_residuals, std_residuals, color='yellow', alpha=0.4, zorder=2)
    plt.title(f'{dataset_name} Data: Histogram of Residuals', fontsize=14, fontweight='bold', )
    plt.xlabel('Residuals (minutes)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.ylim([0,max_count])
    plt.axvline(x=0, color='k', linestyle='--')  # Add a vertical line at x=0 for reference
    plt.text(bins[0], max_count - 20, f'Std Dev = {std_residuals:.2f}\nMAE = {mae:.2f}', fontsize=14, fontweight='bold', verticalalignment='top', horizontalalignment='left')
    # Shading 1 and 2 standard deviation

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('regression_result.pdf')
    plt.show()

# plot the regression results for the test dataset
plot_regression_results(best_model, test_dataset, "Test")


# 5: ANALYSE MULTIHEAD PREDICTIONS (FIGURE 3)
# './logs/logmultihead_720_(32, 64, 128, 256, 256)_None_1.00e-04_1.00e-01:32_42_log.csv'

if load_existing:

    # Read the CSV files into pandas DataFrames
    df1 = pd.read_csv('./logs/logmultihead_720_(32, 64, 128, 256, 256)_None_1.00e-04_1.00e-01:32_42_log.csv', sep=';')

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 2 rows, 1 column for subplots

    # First subplot for loss
    axs[0].plot(df1['epoch'], df1['val_h_loss'], label='hour validation loss', c='blue', linestyle='-')
    axs[0].plot(df1['epoch'], df1['val_m_loss'], label='minute validation loss', c='red', linestyle='-')

    # Customize the first subplot
    axs[0].set_title('Learning curve validation loss', fontsize=18, fontweight='bold', )
    axs[0].set_xlabel('Epochs', fontsize=14, fontweight='bold')
    axs[0].set_ylabel('Loss', fontsize=14, fontweight='bold', )
    axs[0].set_yscale('log')
    axs[0].set_ylim([1.e-4, 10])
    axs[0].set_xlim([0, 150])
    axs[0].tick_params(axis='both', which='major', labelsize=14)
    axs[0].legend(fontsize='large')
    axs[0].grid(True)

    # Second subplot for error
    hour_epoch = 110
    minute_epoch = 139
    axs[1].plot(100 * (1 - df1['val_h_accuracy'][:hour_epoch]), c='blue', label='hour error', linestyle='-')
    axs[1].plot(100 * (1 - df1['val_m_accuracy'][:minute_epoch]), c='red', label='minute error', linestyle='--')

    # Customize the second subplot
    axs[1].set_title('Learning curve classification error', fontsize=18, fontweight='bold')
    axs[1].set_xlabel('Epochs', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('Error (%)', fontsize=14, fontweight='bold')
    axs[1].set_yscale('log')
    axs[1].set_ylim([1.e-2, 100])
    axs[1].set_xlim([0, 150])
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[1].legend(fontsize='large')
    axs[1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined figure as a PDF
    plt.savefig('multihead_combined.pdf')

    # Show the plot
    plt.show()


# 6: LOAD BEST MULTICYCLIC MODEL

# EXISTING DATA
# './best_model/best_model_multicyclic_720_(32, 64, 128, 256, 256)_None_1.00e-04_1.00e-01:32_42/'


if load_existing:

    random_seed = 42

    # data parameters
    mode = 'multicyclic'
    num_images = 18000
    train_split = 0.64   # 80% x 80%
    val_split = 0.16     # 80% x 20%

    # n_categories_list = [120, 144, 180, 240, 360, 720]
    n_categories = 720

    # architecture
    input_shape = (150, 150, 1)
    # n_categories = n_categories_list[4]
    batch_size = 32

    # locations of i/o
    data_dir = './data/CLOCKS/'

    # calculate sizes of splits for later
    num_train = int(train_split * num_images)
    num_val = int(val_split * num_images)
    num_test = num_images - num_train - num_val
    class_names = [str(i) for i in range(n_categories)]

    train_dataset, val_dataset, test_dataset = cu.load_and_process_data(data_dir, num_train, num_val, batch_size, mode, n_categories, random_seed)

    best_model = load_model(f'./best_model/best_model_multicyclic_720_(32, 64, 128, 256, 256)_None_1.00e-04_1.00e-01:32_42/')
    test_images = []
    test_labels = []

    for images, (labels_h, labels_sm, labels_cm) in test_dataset:
        test_images.append(images.numpy())
        stacked_labels = np.stack([labels_h.numpy(), labels_sm.numpy(), labels_cm.numpy()], axis=0)
        test_labels.append(stacked_labels)

    # Once all batches are processed, concatenate them into a single 3 x test_size array
    test_labels = np.concatenate(test_labels, axis=1)

    # Converting list to numpy array for computation
    test_images = np.concatenate(test_images)

    pred_labels = best_model.predict(test_images)

    classification_h = np.argmax(pred_labels[0], axis=-1)
    regression_sm = pred_labels[1].squeeze()  # Remove the singleton dimension
    regression_cm = pred_labels[2].squeeze()  # Remove the singleton dimension

    # Now, all the arrays to be stacked are 1D
    pred_labels = np.stack((classification_h, regression_sm, regression_cm), axis=-1).T


# 7: ANALYSE BEST MULTICYCLIC MODEL

import matplotlib.pyplot as plt
import numpy as np

# convert back to original [h, m] format
# Calculate the angle in radians for hours and minutes from the sine and cosine values
def angles_to_time(angle_labels):

    hours = angle_labels[0, :].astype(int)
    minutes_angle = np.arctan2(angle_labels[1, :], angle_labels[2, :])

    minutes = (minutes_angle * (60 / (2 * np.pi)))  # Convert radian to minute

    # Rounding to the nearest integer
    minutes = np.rint(minutes).astype(int)

    # and make sure we stay in the circle
    minutes = np.mod(np.rint(minutes).astype(int),60).squeeze()

    time_label = np.stack((hours, minutes))

    return time_label

def calculate_r_squared(actual, predicted):
    # Calculate the mean of the actual values
    mean_actual = np.mean(actual)

    # Calculate total sum of squares
    total_sum_of_squares = np.sum((actual - mean_actual) ** 2)

    # Calculate the residual sum of squares
    residual_sum_of_squares = np.sum((actual - predicted) ** 2)

    # Calculate the R² score
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

    return r_squared

def calculate_residual_statistics(actual, predicted):
    residuals = actual - predicted
    std_residuals = np.std(residuals)
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    return residuals, std_residuals, mae, mse, rmse

# convert from (h, sin, cos) => (h, m)
pred_labels_orig = angles_to_time(pred_labels)
test_labels_orig = angles_to_time(test_labels)

# convert from (h, m) => x where x is now a continuous regression variable
pred_labels_line = 60 * pred_labels_orig[0,:] + pred_labels_orig[1,:]
test_labels_line = 60 * test_labels_orig[0,:] + test_labels_orig[1,:]

differences = np.abs(test_labels_line - pred_labels_line)

# Use np.bincount to count occurrences of each difference
# The result will be an array where the index is the difference and the value is the count
difference_counts = np.bincount(differences)
print(difference_counts)

r_squared = calculate_r_squared(test_labels_line, pred_labels_line)
residuals, std_residuals, mae, mse, rmse = calculate_residual_statistics(test_labels_line, pred_labels_line)

print("regression R² score:", r_squared)
print(f"Standard Deviation of Residuals: {std_residuals}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
