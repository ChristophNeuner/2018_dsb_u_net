import os
import numpy as np
import sys
#import cv2
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras import backend as K

smooth = 1.

###
#loads a dataset, resizes it and scales image channels
#datasetPath: path to the dataset
#imgWidth: width the images should be resized to
#imgHeight: height the images should be resized to
#imgChannels: defines if images are rgb or grayscale
#testData: boolean value, if the dataset contains test images and therefore does not contain masks
##
#return:
#ids: image ids
#images: numpy array with images' data
#masks: numpy array with masks' data
#sizes_test: numpy array that contains original sizes of the images to resize the masks for correct submission
###
def load_dataset(datasetPath, imgWidth, imgHeight, imgChannels, testData):
    # Get image IDs
    ids = next(os.walk(datasetPath))[1]    
    #list that saves original sizes of test images
    if testData:        
        sizes_test = []          
    # Get and resize images and masks(if withMasks == true)
    images = np.zeros((len(ids), imgHeight, imgWidth, imgChannels), dtype=np.uint8)
    if not testData:
        masks = np.zeros((len(ids), imgHeight, imgWidth, 1), dtype=np.bool)
    print('Getting and resizing images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        path = datasetPath + id_
        try:
            ###with skimage
            img = imread(path + '/images/' + id_ + '.png')[:,:,:imgChannels]
            ###with cv2
            #img = cv2.imread(path + '/images/' + id_ + '.png')[:,:,:imgChannels]
        except:
            print(id_)
            continue
        if testData:
            sizes_test.append([img.shape[0], img.shape[1]])
        ###resize with skimage.transform
        img = resize(img, (imgHeight, imgWidth), mode='constant', preserve_range=True)
        ###resize with cv2
        #img = cv2.resize(img, (imgHeight, imgWidth), interpolation=cv2.INTER_AREA)
        ###scale image channels
        img = scale_img_channels(img, imgChannels)
        images[n] = img
        if not testData:
            mask = np.zeros((imgHeight, imgWidth, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file, 0)
                #mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
                mask_ = np.expand_dims(resize(mask_, (imgHeight, imgWidth), mode='constant', 
                                          preserve_range=True), axis=-1)
                #mask_ = np.expand_dims(cv2.resize(mask_, (imgHeight, imgWidth), interpolation=cv2.INTER_AREA), axis=-1)
                mask = np.maximum(mask, mask_)
            masks[n] = mask
            
    print('Done!')
    if testData:
        return ids, images, sizes_test
    else:
        return ids, images, masks

def scale_img_channels(an_img, img_channels):
    for i in range(img_channels):
        channel = an_img[:,:,i]
        channel = channel - channel.min()
        channelmax = channel.max()
        if channelmax > 0:
            factor = 255/channelmax
            channel = (channel * factor).astype(int)
        an_img[:,:,i] = channel
        return an_img
    

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred &gt > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Metric function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)