import os
import os.path
import numpy as np
import sys
#import cv2
#from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras import backend as K
from enum import Enum
from unetEnums import DataType

    
###
#if dataset has already been loaded and saved to file, only the files will be loaded
#otherwise loads a dataset, resizes it, scales image channels and saves to file
#datasetPath: path to the dataset
#imgWidth: width the images should be resized to
#imgHeight: height the images should be resized to
#imgChannels: defines if images are rgb or grayscale
#dataType: Enum value, see "DataType" Enum class below
##
#return:
#ids: image ids
#images: numpy array with images' data
#masks: numpy array with masks' data
#sizes_test: numpy array that contains original sizes of the images to resize the masks for correct submission
###
def load_dataset(datasetPath, imgWidth, imgHeight, imgChannels, datasetType):
    try:
        print("trying to load the numpy arrays from binary files")
        if datasetType == DataType.trainData:
            ids = np.load("./BinaryNumpyFiles/stage1_train_fixed_ids.npy").tolist()
            images = np.load("./BinaryNumpyFiles/stage1_train_fixed_images.npy")
            masks = np.load("./BinaryNumpyFiles/stage1_train_fixed_masks.npy")
        if datasetType == DataType.testData:
            ids = np.load("./BinaryNumpyFiles/stage2_test_final_ids.npy").tolist()
            images = np.load("./BinaryNumpyFiles/stage2_test_final_images.npy")
            sizes_test = np.load("./BinaryNumpyFiles/stage2_test_final_sizes_test.npy")
        if datasetType == DataType.valData:
            ids = np.load("./BinaryNumpyFiles/extra_data_ids.npy").tolist()
            images = np.load("./BinaryNumpyFiles/extra_data_images.npy")
            masks = np.load("./BinaryNumpyFiles/extra_data_masks.npy")

    except FileNotFoundError as e:
        print(e)
        # Get image IDs
        ids = next(os.walk(datasetPath))[1]
        #list that saves original sizes of test images
        if datasetType == DataType.testData:        
            sizes_test = []          
        # Get and resize images and masks
        images = np.zeros((len(ids), imgHeight, imgWidth, imgChannels), dtype=np.uint8)
        if datasetType != DataType.testData:
            masks = np.zeros((len(ids), imgHeight, imgWidth, 1), dtype=np.bool)
        print('Getting and resizing images ... ')
        sys.stdout.flush()

        failedIds = []
        for n, id_ in tqdm(enumerate(ids), total=len(ids)):
            path = datasetPath + id_
            try:
                ###with skimage
                img = imread(path + '/images/' + id_ + '.png')[:,:,:imgChannels]
                if datasetType == DataType.testData:
                    sizes_test.append([img.shape[0], img.shape[1]])
                ###with cv2
                #img = cv2.imread(path + '/images/' + id_ + '.png')[:,:,:imgChannels]
            except:
                print("these ids could not be loaded:")
                print(id_)
                failedIds.append(id_)    
                continue
        
            ###resize with skimage.transform
            img = resize(img, (imgHeight, imgWidth), mode='constant', preserve_range=True)
            ###resize with cv2
            #img = cv2.resize(img, (imgHeight, imgWidth), interpolation=cv2.INTER_AREA)
            ###scale image channels
            img = scale_img_channels(img, imgChannels)
            images[n] = img

            if datasetType != DataType.testData:
                mask = np.zeros((imgHeight, imgWidth, 1), dtype=np.bool)
                for mask_file in next(os.walk(path + '/masks/'))[2]:
                    mask_ = imread(path + '/masks/' + mask_file, as_gray = False)
                    #mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
                    mask_ = np.expand_dims(resize(mask_, (imgHeight, imgWidth), mode='constant', 
                                              preserve_range=True), axis=-1)
                    #mask_ = np.expand_dims(cv2.resize(mask_, (imgHeight, imgWidth), interpolation=cv2.INTER_AREA), axis=-1)
                    mask = np.maximum(mask, mask_)
                masks[n] = mask
    
        if(len(failedIds) == 0):
            print('no failed ids')
        else:
            print("removing failed ids")
            for n, failedId in tqdm(enumerate(failedIds), total=len(failedIds)):
                index = ids.index(failedId)
                ids.remove(failedId)
                images = np.delete(images, index, 0)
                if datasetType != DataType.testData:
                    masks = np.delete(masks, index, 0)
        
        if not os.path.exists("./BinaryNumpyFiles/"):
            os.makedirs("./BinaryNumpyFiles/")
        

        #save arrays to disk for faster loading at the next time
        idsAsNpArray = np.asarray(ids)
        if datasetType == DataType.trainData:
            np.save("./BinaryNumpyFiles/stage1_train_fixed_ids.npy", idsAsNpArray)
            np.save("./BinaryNumpyFiles/stage1_train_fixed_images.npy", images)
            np.save("./BinaryNumpyFiles/stage1_train_fixed_masks.npy", masks)
        if datasetType == DataType.testData:
            np.save("./BinaryNumpyFiles/stage2_test_final_ids.npy", idsAsNpArray)
            np.save("./BinaryNumpyFiles/stage2_test_final_images.npy", images)
            np.save("./BinaryNumpyFiles/stage2_test_final_sizes_test.npy", sizes_test)
        if datasetType == DataType.valData:
            np.save("./BinaryNumpyFiles/extra_data_ids.npy", idsAsNpArray)
            np.save("./BinaryNumpyFiles/extra_data_images.npy", images)
            np.save("./BinaryNumpyFiles/extra_data_masks.npy", masks)

    print('Done!')
    if datasetType == DataType.testData:
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
    

# Define IoU metric (probably wrong, better do not use it at the moment)
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
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def binary_crossentropy_with_dice_coef_loss(y_true, y_pred):
    w1 = 1
    w2 = 1
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


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