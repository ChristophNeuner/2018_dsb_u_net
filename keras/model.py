import os
import utils
import tensorflow as tf
import numpy as np
from datetime import datetime
from keras.models import Model, load_model, save_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras import applications
#import cv2
from skimage.transform import resize
from unetEnums import MaskType



def build_unet_inception_resnet_v2(input_shape, numberOfMaskChannels):
    model = get_unet_inception_resnet_v2(input_shape, numberOfMaskChannels)
    model.compile(optimizer='adam', 
                  loss=utils.binary_crossentropy_with_dice_coef_loss, 
                  metrics=[utils.dice_coef, 
                           utils.dice_coef_loss, 
                           utils.binary_crossentropy, 
                           utils.binary_crossentropy_with_dice_coef_loss])
    model.summary()
    return model
   

#Fit model with generator
def fit_model_generator(model, 
                        modelDir, 
                        trainGenerator, 
                        numberOfTrainImages, 
                        valGenerator, 
                        numberOfValImages, 
                        epochs, 
                        batchSize, 
                        maskType):
    earlystopper = EarlyStopping(patience=20, verbose=1)
    if maskType == MaskType.nucleusMask:
        currentModelDir = os.path.join(modelDir, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
    elif maskType == MaskType.spaceBetweenMask:
        currentModelDir = os.path.join(modelDir, datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " spaceBetween",)
    else:
        raise ValueError("maskType must be one of unetEnums.MaskType")
    if not os.path.exists(currentModelDir):
        os.makedirs(currentModelDir)
    filepath = os.path.join(currentModelDir, 'epoch{epoch:04d}-val_loss{val_loss:.2f}.h5')
    checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    tb = TensorBoard(log_dir=currentModelDir, 
                     histogram_freq=0, 
                     batch_size=batchSize, 
                     write_graph=True, 
                     write_grads=False, 
                     write_images=False, 
                     embeddings_freq=0, 
                     embeddings_layer_names=None, 
                     embeddings_metadata=None)   
    rlop = ReduceLROnPlateau(monitor='val_loss', 
                             factor= 0.5, 
                             patience= 1, 
                             verbose= 1, 
                             mode= 'auto', 
                             epsilon= 0.0001, 
                             cooldown= 1, 
                             min_lr= 1e-7)
    results = model.fit_generator(generator=trainGenerator, 
                                  validation_data=valGenerator, 
                                  steps_per_epoch=numberOfTrainImages // batchSize, 
                                  validation_steps = numberOfValImages // batchSize, 
                                  epochs=epochs, workers=4, 
                                  callbacks=[earlystopper, checkpointer, tb, rlop])
    

#Make predictions
#Predict on train, val and test
def make_predictions(model_path, X_train, X_val, X_test, maskType):
    if maskType==MaskType.nucleusMask:
        t = 0.5
    elif maskType==MaskType.spaceBetweenMask:
        t = 0.2
    else:
        raise ValueError("maskType must be one of unetEnums.MaskType")

    model = load_model(model_path, custom_objects={'dice_coef': utils.dice_coef, 
                                                   'dice_coef_loss':utils.dice_coef_loss, 
                                                   'binary_crossentropy':utils.binary_crossentropy, 
                                                   'binary_crossentropy_with_dice_coef_loss':utils.binary_crossentropy_with_dice_coef_loss})
    #preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
    #preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
    preds_train = model.predict(X_train, verbose=1)
    preds_val = model.predict(X_val, verbose=1)
    preds_test = model.predict(X_test, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > t).astype(np.uint8)
    preds_val_t = (preds_val > t).astype(np.uint8)
    preds_test_t = (preds_test > t).astype(np.uint8)

    return preds_train, preds_val, preds_test, preds_train_t, preds_val_t, preds_test_t

def upsamplePredictionsToOriginalSize(predsTest, originalSizes):
    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(predsTest)):
        ###skimage
        preds_test_upsampled.append(resize(np.squeeze(predsTest[i]), 
                                           (originalSizes[i][0], originalSizes[i][1]), 
                                           mode='constant', preserve_range=True))
        ###cv2
        #preds_test_upsampled.append(cv2.resize(np.squeeze(predsTest[i]),
                                                #(originalSizes[i][0], originalSizes[i][1]),
                                                #interpolation=cv2.INTER_AREA))

    return preds_test_upsampled



### mostly from https://github.com/killthekitten/kaggle-carvana-2017/blob/master/models.py

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from inception_resnet_v2 import InceptionResNetV2
#from params import args


def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

"""
Unet with Inception Resnet V2 encoder
Uses the same preprocessing as in Inception, Xception etc. (imagenet_utils.preprocess_input with mode 'tf' in new Keras version)
"""
def get_unet_inception_resnet_v2(input_shape, numberOfMaskChannels):
    if numberOfMaskChannels == 1:
        base_model = GetOrBuildModel("./untrained_models/inception_resnet_v2_model_untrained_one_channel_masks_notop.h5")
    elif numberOfMaskChannels == 2:
        base_model = GetOrBuildModel("./untrained_models/inception_resnet_v2_model_untrained_two_channel_masks_notop.h5")
    else:
        raise ValueError('numberOfMaskChannels must be 1 or 2')

    conv1 = base_model.get_layer('activation_3').output
    conv2 = base_model.get_layer('activation_5').output
    conv3 = base_model.get_layer('block35_10_ac').output
    conv4 = base_model.get_layer('block17_20_ac').output
    conv5 = base_model.get_layer('conv_7b_ac').output
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.4)(conv10)
    if numberOfMaskChannels == 1:
        x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    elif numberOfMaskChannels == 2:
        x = Conv2D(2, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model


def GetOrBuildModel(modelPath):
    if(os.path.isfile(modelPath)):
        base_model = load_model(modelPath)
    else:
        base_model = InceptionResNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
        save_model(base_model, modelPath, overwrite=True)
        
    return base_model



####
############### obsolete ######################
####

# Build original U-Net model: https://arxiv.org/abs/1505.04597
def build_original_unet(imgHeight, imgWidth, imgChannels):   
    inputs = Input((imgHeight, imgWidth, imgChannels))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', 
                  loss=utils.binary_crossentropy_with_dice_coef_loss, 
                  metrics=[utils.dice_coef, 
                           utils.dice_coef_loss, 
                           utils.binary_crossentropy, 
                           utils.binary_crossentropy_with_dice_coef_loss])
    model.summary()
    return model


#Fit model
def fit_model(model, modelDir, X_train, Y_train, validationSplit, epochs, batchSize, maskType):
    earlystopper = EarlyStopping(patience=20, verbose=1)
    if maskType == MaskType.nucleusMask:
        currentModelDir = os.path.join(modelDir, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
    elif maskType == MaskType.spaceBetweenMask:
        currentModelDir = os.path.join(modelDir, datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " spaceBetween",)
    else:
        raise ValueError("maskType must be one of unetEnums.MaskType")
    if not os.path.exists(currentModelDir):
        os.makedirs(currentModelDir)
    filepath = os.path.join(currentModelDir, 'epoch{epoch:04d}-val_loss{val_loss:.2f}.h5')
    checkpointer = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    tb = TensorBoard(log_dir=currentModelDir, histogram_freq=0, batch_size=batchSize, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)   
    rlop = ReduceLROnPlateau(monitor='val_loss', factor= 0.5, patience= 1, verbose= 1, mode= 'auto', epsilon= 0.0001, cooldown= 1, min_lr= 1e-7)
    results = model.fit(X_train, Y_train, validation_split=validationSplit, batch_size=batchSize, epochs=epochs, 
                        callbacks=[earlystopper, checkpointer, rlop, tb])