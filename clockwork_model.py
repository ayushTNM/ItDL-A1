import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Lambda
from keras.layers import Flatten, Dropout, Dense


def create_model(mode, input_shape, n_categories, num_kernels=(32, 64, 128, 256, 256), depth=None, dropout1=0.5, dropout2=0.25):
    inputs = Input(shape=input_shape)

    # First convolutional block
    x = Conv2D(num_kernels[0], kernel_size=(7, 7), activation='relu', name='conv1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    
    # Second convolutional block
    x = Conv2D(num_kernels[1], (5, 5), activation='relu', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)

    # Third convolutional block - added to further reduce feature dimensions
    x = Conv2D(num_kernels[2], (5, 5), activation='relu', name='conv3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)

    if depth == 3:
        x = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)

    # Fourth convolutional block - added to further reduce feature dimensions
    x = Conv2D(num_kernels[3], (5, 5), activation='relu', name='conv4')(x)
    x = MaxPooling2D((2, 2), name='pool4')(x)

    if depth == 4:
        x = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)

    # Fifth convolutional block - added to further reduce feature dimensions
    x = Conv2D(num_kernels[4], (3, 3), activation='relu', name='conv5')(x)
    x = MaxPooling2D((2, 2), name='pool5')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout1, name='do_1')(x)
    x = BatchNormalization(name='bnf')(x)
    
    if mode == 'classify':

        x = Dense(512, activation='relu', name='d1')(x)
        x = Dropout(dropout2, name='do_2')(x)

        x = Dense(512, activation='relu', name='d2')(x)
    
        x = Dense(n_categories, activation='softmax', name='t')(x)
    
        x = Model(inputs=inputs, outputs=x)

    # model 2
    elif mode == 'regress':

        x = Dense(512, activation='relu', name='d1')(x)
        x = Dropout(dropout2, name='do_2')(x)

        x = Dense(512, activation='relu', name='d2')(x)

        x = Dense(1, activation=None, name='t')(x)  
        
        x = Model(inputs=inputs, outputs=x)

    # model 3
    elif mode == 'multihead':
       
        h = Dense(512, activation='relu', name='hd1')(x)
        h = Dropout(dropout2, name='do_h2')(h)
        h = Dense(512, activation='relu', name='hd2')(h)
        h = Dense(12, activation='softmax', name='h')(h)

        m = Dense(512, activation='relu', name='md1')(x)
        m = Dropout(dropout2, name='do_m2')(m)
        m = Dense(512, activation='relu', name='md2')(m)
        m = Dense(int(n_categories/12), activation='softmax', name='m')(m)

        x = Model(inputs=inputs, outputs=[h, m])

    elif mode == 'cyclic':

        sh = Dense(512, activation='tanh', name='sh1')(x)
        sh = Dropout(dropout2, name='do_sh2')(sh)
        sh = Dense(512, activation='tanh', name='sh2')(sh)
        sh = Dense(1, activation=None, name='sh')(sh)

        ch = Dense(512, activation='tanh', name='ch1')(x)
        ch = Dropout(dropout2, name='do_ch2')(ch)
        ch = Dense(512, activation='tanh', name='ch2')(ch)
        ch = Dense(1, activation=None, name='ch')(ch)
        
        sm = Dense(512, activation='tanh', name='sm1')(x)
        sm = Dropout(dropout2, name='do_sm2')(sm)
        sm = Dense(512, activation='tanh', name='sm2')(sm)
        sm = Dense(1, activation=None, name='sm')(sm)

        cm = Dense(512, activation='tanh', name='cm1')(x)
        cm = Dropout(dropout2, name='do_cm2')(cm)
        cm = Dense(512, activation='tanh', name='cm2')(cm)
        cm = Dense(1, activation=None, name='cm')(cm)

        x = Model(inputs=inputs, outputs=[sh, ch, sm, cm])

    elif mode == 'multicyclic':

        h = Dense(512, activation='relu', name='hd1')(x)
        h = Dropout(dropout2, name='do_h2')(h)
        h = Dense(512, activation='relu', name='hd2')(h)
        h = Dense(12, activation='softmax', name='h')(h)
        
        sm = Dense(512, activation='tanh', name='sm1')(x)
        sm = Dropout(dropout2, name='do_sm2')(sm)
        sm = Dense(512, activation='tanh', name='sm2')(sm)
        sm = Dense(1, activation=None, name='sm')(sm)

        cm = Dense(512, activation='tanh', name='cm1')(x)
        cm = Dropout(dropout2, name='do_cm2')(cm)
        cm = Dense(512, activation='tanh', name='cm2')(cm)
        cm = Dense(1, activation=None, name='cm')(cm)

        x = Model(inputs=inputs, outputs=[h, sm, cm])

    else:

        print("invalid model")

        exit(0)
    
    return x

