#!/usr/bin/env python
# coding: utf-8

# 1: LOAD LIBS, UTILITIES, DATASETS
# 2A: MLP DEFINE AUTO-TUNER
# 2B: MLP OPTION 1: CREATE NEW AUTO-TUNER RESULTS (TAKES > 1 HOUR)
# 2C: MLP OPTION 2: LOAD EXISTING AUTO-TUNER RESULTS
# 2D: MLP APPLY 3 BEST MLP MODELS TO CIFAR10 DATA (overwrites existing results)
# 2E: MLP COMPARE RESULTS WITH CIFAR10 DATA
# 3A: CNN DEFINE AUTO-TUNER
# 3B: CNN OPTION 1: CREATE NEW AUTO-TUNER RESULTS (TAKES > 1 HOUR)
# 3C: CNN OPTION 2: LOAD EXISTING AUTO-TUNER RESULTS
# 3D: CNN APPLY 3 BEST MLP MODELS TO CIFAR10 DATA (overwrites existing results)
# 3E: CNN COMPARE RESULTS WITH CIFAR10 DATA

# set to False if you want to run new data, True if existing data is available
load_existing = True

# LOAD LIBS, UTILITIES, DATASETS

# ! pip install keras_tuner
import numpy as np
import os
import datetime
import keras
import tensorflow as tf
from keras.datasets import fashion_mnist, cifar10
from keras import layers
import keras_tuner
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# no GPU available
tf.config.set_visible_devices([], 'GPU')
optimizers = {
    "sgd": keras.optimizers.SGD,
    "adam": keras.optimizers.Adam,
}

# set up callbacks
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss',factor=0.6,patience=2,min_lr=1.0e-9)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
callbacks=[early_stopping, tensorboard, lr_schedule]

# LOAD & SHOW FASHION MNIST DATA
fm_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

num_classes = len(fm_class_names)
(fm_x_train,fm_y_train), (fm_x_test,fm_y_test) = fashion_mnist.load_data()
fm_x_train, fm_x_test = fm_x_train/255.0, fm_x_test/255.0
fm_y_train_cat = keras.utils.to_categorical(fm_y_train, num_classes)
fm_y_test_cat = keras.utils.to_categorical(fm_y_test, num_classes)
print(fm_y_train)

# Create a 3x4 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

# Plot 3x4 images from x_train
for ind, ax in enumerate(axes.ravel()):
    ax.imshow(fm_x_train[ind], cmap='gray')
    ax.set_title(f"Label: {fm_class_names[fm_y_train[ind]]}")
    ax.axis('off')

# Display the subplots
plt.tight_layout()
plt.show()

# LOAD & SHOW CFAR10 DATA
cf_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(cf_x_train,cf_y_train), (cf_x_test,cf_y_test) = cifar10.load_data()
cf_x_train, cf_x_test = cf_x_train/255.0, cf_x_test/255.0
cf_y_train, cf_y_test = cf_y_train.ravel(), cf_y_test.ravel()
cf_y_train_cat = keras.utils.to_categorical(cf_y_train, num_classes)
cf_y_test_cat = keras.utils.to_categorical(cf_y_test, num_classes)
print(cf_y_train)

# Create a 3x4 grid of subplots
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

# Plot 3x4 images from x_train
for ind, ax in enumerate(axes.ravel()):
    ax.imshow(cf_x_train[ind])
    ax.set_title(f"Label: {cf_class_names[cf_y_train[ind]]}")
    ax.axis('off')

# Display the subplots
plt.tight_layout()
plt.show()


# 2A: MLP DEFINE AUTO-TUNER

def build_model_mlp(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    activation=hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    # Tune the number of layers.
    for i in range(3):
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=hp.Int(f"units_{i}", min_value=num_classes*2, max_value=num_classes*50, step=num_classes*2),
                activation=activation,
            )
        )
        if dropout:
            model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(10, activation="softmax"))
    optimizer_choice = hp.Choice("optimizer", ["sgd","adam"])
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")

    optimizer = optimizers[optimizer_choice](learning_rate = learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# 2B: MLP OPTION 1: CREATE NEW AUTO-TUNER RESULTS (TAKES > 1 HOUR)
if load_existing == False:
    dir_name, project_name = "mlp_dir","mlp"

    tuner_mlp = keras_tuner.BayesianOptimization(
        hypermodel=build_model_mlp,
        objective="val_accuracy",
        max_trials=40,
        num_initial_points=10,  # Number of initial random trials
        overwrite=True,
        directory=dir_name,
        project_name=project_name,
    )
    tuner_mlp.search_space_summary()

    # Setting up TensorBoard callback

    tuner_mlp.search(fm_x_train, fm_y_train_cat, epochs=30, validation_split=0.1, callbacks=callbacks)

    best_models = tuner_mlp.get_best_models(num_models=10)
    best_hps = tuner_mlp.get_best_hyperparameters(num_trials=10)
    tuner_mlp.results_summary(num_trials=10)


# 2C: MLP OPTION 2: LOAD EXISTING AUTO-TUNER RESULTS
if load_existing == True:
    dir_name, project_name = f"mlp","mlp"

    tuner_mlp = keras_tuner.BayesianOptimization(
        hypermodel=build_model_mlp,
        objective="val_accuracy",
        max_trials=40,
        num_initial_points=10,  # Number of initial random trials
        overwrite=False,
        directory=dir_name,
        project_name=project_name
    )

    print(tuner_mlp.results_summary(num_trials=3))


# 2D: APPLY 3 BEST MLP MODELS TO CIFAR10 DATA

if load_existing == False:

    filename = "cifar10_mlp_best_hp"
    if not os.path.exists(filename):
        os.makedirs(filename)
        best_hp_sets = tuner_mlp.get_best_hyperparameters(num_trials=3)

    for ind,best_hps in enumerate(best_hp_sets):
        model_best_cf = build_model_mlp(best_hps)
        model_best_cf.build(cf_x_train.shape)
        print(model_best_cf.summary())
        history = model_best_cf.fit(cf_x_train, cf_y_train_cat, epochs=30, validation_split=0.1, callbacks=callbacks)
        np.save(f'{filename}/{filename}_{ind+1}.npy', history.history)  # Save the history as a NumPy array


# 2E: COMPARE RESULTS WITH CIFAR10 DATA
if load_existing == True:

  for ind,best_trial in enumerate(tuner_mlp.oracle.get_best_trials(3)):
    cf_hist = np.load(f"cifar10_mlp_best_hp/cifar10_mlp_best_hp_{ind+1}.npy",allow_pickle=True).item()
    best_cf_val_acc_ind = np.argmax(cf_hist["val_accuracy"])
    print(f"Best trial {ind+1}")
    print(f'fashion_mnist accuracy: {best_trial.metrics.get_best_value("accuracy"):.3f}, fashion_mnist val_accuracy:{best_trial.metrics.get_best_value("val_accuracy"):.3f}')
    print(f'cifar10 accuracy: {cf_hist["accuracy"][best_cf_val_acc_ind]:.3f}, cifar10 val_accuracy:{cf_hist["val_accuracy"][best_cf_val_acc_ind]:.3f}\n')


#3A: DEFINE AUTO-TUNER CNN

def build_model_cnn(hp,input_shape=(28,28,1)):
    model = keras.Sequential()
    activation=hp.Choice("activation", ["relu", "tanh"])
    inp_filters = hp.Int("input_conv_filter", min_value=32, max_value = 128, step=32)
    model.add(layers.Conv2D(
        filters=inp_filters,
        kernel_size=hp.Choice('input_conv_kernel', values = [5,7]),
        activation=activation,
        input_shape=input_shape)
    )
    model.add(layers.MaxPooling2D(2))
    filters = hp.Int("conv_filters", min_value=inp_filters, max_value=256, step=32)
    for i in range(hp.Int("conv_layers",0,2)):
      model.add(
          layers.Conv2D(
              # Tune number of units separately.
              filters=filters,
              kernel_size=3,
              activation=activation,
          )
      )
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    dropout = hp.Boolean("dropout")
    # Tune the number of layers.
    unit_min, unit_max= 128,256
    for i in range(hp.Int("num_layers",1,2)):
        unit_count = hp.Int(f"units_{i}", min_value=unit_min, max_value=unit_max, step=32)
        model.add(
            layers.Dense(
                # Tune number of units separately.
                units=unit_count,
                activation=activation,
            )
        )
        if dropout:
            model.add(layers.Dropout(rate=0.25))
        unit_min, unit_max = int(unit_min/2), unit_count

    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-2, max_value=1e-1, sampling="log")
    optimizer = optimizers[hp.Choice("optimizer",['sgd','adam'])]()
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


#3B: OPTION 1: CREATE NEW AUTO-TUNER RESULTS (TAKES > 1 HOUR)
if load_existing == False:

    dir_name, project_name = "cnn_dir","cnn"

    tuner_cnn = keras_tuner.BayesianOptimization(
        hypermodel=build_model_cnn,
        objective="val_accuracy",
        max_trials=20,
        num_initial_points=10,  # Number of initial random trials
        overwrite=True,
        directory=dir_name,
        project_name=project_name,
    )
    tuner_cnn.search_space_summary()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss',factor=0.6,patience=2,min_lr=1.0e-9)
    tuner_cnn.search(fm_x_train, fm_y_train_cat, epochs=20, validation_split=0.1,callbacks=callbacks)


#3C: OPTION 2: LOAD EXISTING AUTO-TUNER RESULTS
if load_existing == True:

    dir_name, project_name = f"cnn","cnn"
    # loaded_tuner = keras_tuner. from_directory('content/test')
    tuner_cnn = keras_tuner.BayesianOptimization(
        hypermodel=build_model_cnn,
        objective="val_accuracy",
        max_trials=40,
        num_initial_points=10,  # Number of initial random trials
        overwrite=False,
        directory=dir_name,
        project_name=project_name
    )
    print(tuner_cnn.results_summary(num_trials=3))


# 3D: APPLY 3 BEST CNN MODELS TO CIFAR10 DATA
if load_existing == False:

   filename = "cifar10_cnn_best_hp"
   if not os.path.exists(filename):
      os.makedirs(filename)

   best_hp_sets = tuner_cnn.get_best_hyperparameters(num_trials=3)

   for ind,best_hps in enumerate(best_hp_sets):
      model_best_cf = build_model_cnn(best_hps,input_shape=[cf_x_train.shape[1],cf_x_train.shape[2],cf_x_train.shape[3]])
      print(cf_x_train.shape)
      model_best_cf.build()
      print(model_best_cf.summary())
      history = model_best_cf.fit(cf_x_train, cf_y_train_cat, epochs=20, validation_split=0.1, callbacks=callbacks)
      # np.save(f'{filename}/{filename}_{ind+1}.npy', history.history)  # Save the history as a NumPy array
      plt.plot(history.history['val_accuracy'], label=f'Model {ind+1}')

   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.yscale('log')
   plt.grid(True, which='both')
   plt.ylim([0.2,1])
   plt.xlim([0,20])
   plt.legend()
   plt.tight_layout()
   plt.savefig('./figures/CIFAR10_CNN.pdf')
   plt.show()


# 3E: COMPARE RESULTS WITH CIFAR10 DATA

for ind,best_trial in enumerate(tuner_cnn.oracle.get_best_trials(3)):
  cf_hist = np.load(f"cifar10_cnn_best_hp/cifar10_cnn_best_hp_{ind+1}.npy",allow_pickle=True).item()
  best_cf_val_acc_ind = np.argmax(cf_hist["val_accuracy"])
  print(f"Best trial {ind+1}")
  print(f'fashion_mnist accuracy: {best_trial.metrics.get_best_value("accuracy"):.3f}, fashion_mnist val_accuracy:{best_trial.metrics.get_best_value("val_accuracy"):.3f}')
  print(f'cifar10 accuracy: {cf_hist["accuracy"][best_cf_val_acc_ind]:.3f}, cifar10 val_accuracy:{cf_hist["val_accuracy"][best_cf_val_acc_ind]:.3f}\n')


import matplotlib.pyplot as plt
import numpy as np

def plot_images(correct_images, incorrect_images, correct_labels, incorrect_labels, pred_labels_correct, pred_labels_incorrect, class_names):
    """Plot images with their class names, 5 correct on top and 5 incorrect on bottom."""
    plt.figure(figsize=(12, 5))  # Increased figure size for two rows

    # Plot the first 5 correctly classified images
    for i in range(5):
        plt.subplot(2, 5, i+1)  # Changed to 2 rows and 5 columns
        plt.imshow(correct_images[i], interpolation='none')
        true_class = class_names[correct_labels[i]]
        pred_class = class_names[pred_labels_correct[i]]
        plt.title(f"True: {true_class}\nPred: {pred_class}")
        plt.axis('off')

    # Plot the first 5 incorrectly classified images
    for i in range(5):
        plt.subplot(2, 5, i+6)  # Positions 6-10 for the bottom row
        plt.imshow(incorrect_images[i], interpolation='none')
        true_class = class_names[incorrect_labels[i]]
        pred_class = class_names[pred_labels_incorrect[i]]
        plt.title(f"True: {true_class}\nPred: {pred_class}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./figures/CNN_transfer_CIFAR.pdf')
    plt.show()

# Load your dataset
test_images, test_labels_cat = cf_x_test, cf_y_test_cat

# Get predictions
predictions = model_best_cf.predict(test_images)

# Convert predictions to labels
pred_labels = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_labels_cat, axis=1)

# Compare predictions to true labels
correct = pred_labels == test_labels
incorrect = ~correct

# Get images and labels for correct classifications
correct_images = test_images[correct]
correct_labels = test_labels[correct]

# Get images and labels for incorrect classifications
incorrect_images = test_images[incorrect]
incorrect_labels = test_labels[incorrect]

cf_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plot_images(
    correct_images[:5], incorrect_images[:5],
    correct_labels[:5], incorrect_labels[:5],
    pred_labels[correct][:5], pred_labels[incorrect][:5],
    cf_class_names
)

print(np.sum(incorrect))


test_images, test_labels = cf_x_test, cf_y_test_cat
print(test_labels)
