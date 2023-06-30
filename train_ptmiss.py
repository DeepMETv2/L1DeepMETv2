#!/usr/bin/env python
# coding: utf-8

# modified from Jan and Markus's code

import os
import pathlib
import datetime
import h5py
import optparse
import numpy as np

from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Flatten, Reshape, Dense, BatchNormalization, Concatenate, Embedding
from keras import optimizers, initializers
from keras.layers import Lambda
from keras.backend import slice

import tensorflow as tf
import keras.backend as K

from tensorflow import train

# Local imports
from cyclical_learning_rate import CyclicLR
from weighted_sum_layer import weighted_sum_layer
from utils import preProcessing, plot_history, read_input
from loss import custom_loss

def create_model(n_features=8, n_features_cat=3, n_dense_layers=3, activation='tanh', with_bias=False):
    # continuous features
    # [b'PF_dxy', b'PF_dz', b'PF_eta', b'PF_mass', b'PF_puppiWeight', b'PF_charge', b'PF_fromPV', b'PF_pdgId',  b'PF_px', b'PF_py']
    inputs_cont = Input(shape=(maxNPF, n_features), name='input')
    pxpy = Lambda(lambda x: slice(x, (0, 0, n_features-2), (-1, -1, -1)))(inputs_cont)

    embeddings = []
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(maxNPF, 1), name='input_cat{}'.format(i_emb))
        if i_emb == 0:
            inputs = [inputs_cont, input_cat]
        else:
            inputs.append(input_cat)
        embedding = Embedding(input_dim=emb_input_dim[i_emb], output_dim=emb_out_dim, embeddings_initializer=initializers.RandomNormal(mean=0., stddev=0.4/emb_out_dim), name='embedding{}'.format(i_emb))(input_cat)
        embedding = Reshape((maxNPF, 8))(embedding)
        embeddings.append(embedding)

    x = Concatenate()([inputs[0]] + [emb for emb in embeddings])

    for i_dense in range(n_dense_layers):
        x = Dense(8*2**(n_dense_layers-i_dense), activation=activation, kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)

    # List of weights. Increase to 3 when operating with biases
    # Expect typical weights to not be of order 1 but somewhat smaller, so apply explicit scaling
    x = Dense(3 if with_bias else 1, activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
    #print('Shape of last dense layer', x.shape)

    x = Concatenate()([x, pxpy])
    x = weighted_sum_layer(with_bias, name = "weighted_sum" if with_bias else "output")(x)

    if with_bias:
        x = Dense(2, activation='linear', name='output')(x)

    outputs = x 
    return inputs, outputs


# configuration
usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-i', '--input', dest='input',
                  help='input file', default='tree_100k.h5', type='string')
parser.add_option('-l', '--load', dest='load',
                  help='load model from timestamp', default='', type='string')
parser.add_option('--nfiles', dest='nfiles', 
                  help='number of h5df files', default=100, type='int')
parser.add_option('--withbias', dest='withbias',
                  help='include bias term in the DNN', default=False, action="store_true")
(opt, args) = parser.parse_args()

# general setup
maxNPF = 4500
n_features_pf = 8
n_features_pf_cat = 3
normFac = 50.
epochs = 100
batch_size = 64
preprocessed = True
emb_out_dim = 8


##
## read input and do preprocessing
##
Xorg, Y = read_input(opt.input)
Y = Y / -normFac

Xi, Xc1, Xc2, Xc3 = preProcessing(Xorg)
print(Xc1.dtype)
Xc = [Xc1, Xc2, Xc3]
emb_input_dim = {
    i:int(np.max(Xc[i][0:1000])) + 1 for i in range(n_features_pf_cat)
}
print('Embedding input dimensions', emb_input_dim)

# prepare training/val data
Yr = Y
Xr = [Xi] + Xc
indices = np.array([i for i in range(len(Yr))])
indices_train, indices_test = train_test_split(indices, test_size=0.2, random_state=7)

Xr_train = [x[indices_train] for x in Xr]
Xr_test = [x[indices_test] for x in Xr]
Yr_train = Yr[indices_train]
Yr_test = Yr[indices_test]

# inputs, outputs = create_output_graph()
inputs, outputs = create_model(n_features=n_features_pf, n_features_cat=n_features_pf_cat, with_bias=opt.withbias)

lr_scale = 1.
clr = CyclicLR(base_lr=0.0003*lr_scale, max_lr=0.001*lr_scale, step_size=len(Y)/batch_size, mode='triangular2')

# create the model
model = Model(inputs=inputs, outputs=outputs)
optimizer = optimizers.Adam(lr=1., clipnorm=1.)
model.compile(loss=custom_loss, optimizer=optimizer, 
               metrics=['mean_absolute_error', 'mean_squared_error'])
model.summary()

if opt.load:
    timestamp = opt.load
else:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
path = f'models/{timestamp}'
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

plot_model(model, to_file=f'{path}/model.png', show_shapes=True)

if opt.load:
    model.load_weights(f'{path}/model.h5')
    print(f'Restored model {timestamp}')

with open(f'{path}/summary.txt', 'w') as txtfile:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: txtfile.write(x + '\n'))

# early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

csv_logger = CSVLogger(f"{path}/loss_history.csv")

# model checkpoint callback
# this saves our model architecture + parameters into model.h5
model_checkpoint = ModelCheckpoint(f'{path}/model.h5', monitor='val_loss',
                                   verbose=0, save_best_only=True,
                                   save_weights_only=False, mode='auto',
                                   period=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=0.000001, cooldown=3, verbose=1)

stop_on_nan = keras.callbacks.TerminateOnNaN()


# Run training
history = model.fit(Xr_train, 
                    Yr_train,
                    epochs=epochs,
                    verbose=1,  # switch to 1 for more verbosity
                    validation_data=(Xr_test, Yr_test),
                    callbacks=[early_stopping, clr, stop_on_nan, csv_logger],#, reduce_lr], #, lr,   reduce_lr],
                   )

# Plot loss
plot_history(history, path)

model.save(f'{path}/model.h5')
from tensorflow import saved_model
saved_model.simple_save(K.get_session(), f'{path}/saved_model', inputs={t.name:t for t in model.input}, outputs={t.name:t for t in model.outputs})
