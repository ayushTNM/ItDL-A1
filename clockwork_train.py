import os
import csv
import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import clockwork_model as cw
import clockwork_utils as cu

def main(num_kernels, learning_rate, depth, dropout, mode, n_categories):
    
    random_seed = 42

    # data parameters
    num_images = 18000
    train_split = 0.64   # 80% x 80%
    val_split = 0.16     # 80% x 20%
    test_split = 1 - train_split - val_split

    #architecture
    input_shape = (150, 150, 1)
    n_categories = n_categories
    num_kernels=num_kernels
    depth=depth
    dropout1=dropout
    dropout2=dropout/2
    learning_rate = learning_rate

    # hyper parameters
    mode = mode
    batch_size = 32
    epochs = 200
    patience = 6

    # locations of i/o
    data_dir = f'./data/CLOCKS/'  
    marker_name = f'{mode}_{n_categories}_{num_kernels}_{depth}_{learning_rate:.2e}_{dropout1:.2e}:{batch_size}_{random_seed}'
    best_model_dir = './best_model'
    best_model_path = best_model_dir + f'/best_model_{marker_name}' 
    checkpoint_dir = f'./training/training_{marker_name}' 

    log_dir = cu.create_log_dir(best_model_dir, marker_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    # calculate sizes of splits for later
    num_train = int(train_split * num_images)
    num_val = int(val_split * num_images)
    num_test = num_images - num_train - num_val  

    print(f"loading from {data_dir}")
    train_dataset, val_dataset, test_dataset = cu.load_and_process_data(data_dir, num_train, num_val, batch_size, mode, n_categories, random_seed)
    
    # for images, labels in test_dataset.take(1):
    #     print(labels.numpy())          

    # [Model loading or creation] 
    model = cw.create_model(
        mode=mode,
        input_shape=input_shape, 
        n_categories=n_categories,
        num_kernels=num_kernels,
        depth=depth,
        dropout1=dropout1,
        dropout2=dropout2)

    compiler_loss, compiler_metrics, compiler_weights = cu.get_compiler(mode)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=compiler_loss,
        loss_weights=compiler_weights,
        metrics=compiler_metrics)  

    model.summary()

    callbacks_list = cu.get_callbacks(log_dir, checkpoint_path, patience, mode)

    model.fit(
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

    test_results = best_model.evaluate(test_dataset, steps=num_test // batch_size) 

    print(f'test dataset:')
    print(f'  loss : {test_results[0]}')
    print(f'metric : {test_results[1]}')

    return 

if __name__ == "__main__":
    # select key parameters here
    # time is in 720 categories, can be grouped together by choosing smaller n_categories 
    # n_categories_list = [120, 144, 180, 240, 360, 720]
    n_categories_list = [720]
    # different models and labels, can run in batch or just select one model
    mode_list=['classify', 'regress', 'multihead', 'cyclic', 'multicyclic']
    mode = mode_list[1]
    # kernels of convolution layers that worked the best
    num_kernels_list=[(32, 64, 128, 256, 256)]
    # depth layer to extract scaling/rotation. Did not outperform
    depth_list=[None]
    # select different learning rates to trial
    lr_list = [1.e-4]
    # drop-out rate
    dropout_list = [0.012]
    # not used now, but keep in reserve if many models are used and you want to label them
    global_index = 0  
    
    for nk_idx, num_kernels in enumerate(num_kernels_list):
        for d_idx, depth in enumerate(depth_list):
            for lr_idx, lr in enumerate(lr_list):        
                    for do_idx, dropout in enumerate(dropout_list):
                        for nc_idx, n_categories in enumerate(n_categories_list):
                            results = main(num_kernels, lr, depth, dropout, mode, n_categories)
                            print(f"model {global_index}: number of categories {n_categories} kernel = {num_kernels} depth = {depth}, learning rate = {lr}, dropout = {dropout}")
                            global_index += 1