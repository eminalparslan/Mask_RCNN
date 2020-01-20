from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn import utils
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
from os import listdir
from xml.etree import ElementTree
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import math
import numpy as np
import imgaug


class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"

    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # number of classes (we would normally add +1 for the background)
    NUM_CLASSES = 6 + 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300

    # Learning rate
    LEARNING_RATE = 0.001

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.3

    # setting Max ground truth instances
    MAX_GT_INSTANCES = 10

    # BACKBONE                       resnet101
    # BACKBONE_STRIDES               [4, 8, 16, 32, 64]
    # BATCH_SIZE                     1
    # BBOX_STD_DEV                   [0.1 0.1 0.2 0.2]
    # COMPUTE_BACKBONE_SHAPE         None
    DETECTION_MAX_INSTANCES = 10
    # DETECTION_NMS_THRESHOLD        0.3
    # FPN_CLASSIF_FC_LAYERS_SIZE     1024
    # GPU_COUNT                      1
    # GRADIENT_CLIP_NORM             5.0
    # IMAGES_PER_GPU                 1
    # IMAGE_CHANNEL_COUNT            3

    IMAGE_MAX_DIM = 128
    # IMAGE_META_SIZE                19
    IMAGE_MIN_DIM = 128
    # IMAGE_MIN_SCALE                0
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_SHAPE = [132, 132, 3]

    # LEARNING_MOMENTUM              0.9
    #LEARNING_RATE = 0.003
    # LOSS_WEIGHTS                   {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 1.0}
    # MASK_POOL_SIZE                 14
    # MASK_SHAPE                     [28, 28]
    # MAX_GT_INSTANCES               10

    # MEAN_PIXEL = [212.0, 47.0, 17.0]  # for
    MEAN_PIXEL = [24.9, 128.3, 235.6]  # for D1
    # MINI_MASK_SHAPE                (56, 56)
    # NAME                           MaskRCNN_config
    # NUM_CLASSES                    7
    # POOL_SIZE                      7

    POST_NMS_ROIS_INFERENCE = 10
    POST_NMS_ROIS_TRAINING = 200

    PRE_NMS_LIMIT = 600
    # ROI_POSITIVE_RATIO             0.33
    #RPN_ANCHOR_RATIOS = [1]

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # RPN_ANCHOR_STRIDE              1
    # RPN_BBOX_STD_DEV               [0.1 0.1 0.2 0.2]
    # RPN_NMS_THRESHOLD              0.7

    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    # STEPS_PER_EPOCH                131
    # TOP_DOWN_PYRAMID_SIZE          256
    # TRAIN_BN                       False
    TRAIN_ROIS_PER_IMAGE = 66
    USE_MINI_MASK = False
    # USE_RPN_ROIS                   True
    VALIDATION_STEPS = 50
# WEIGHT_DECAY                   0.0001


# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "galaxy inference"
    # number of classes (background + 6 galaxy classes)
    NUM_CLASSES = 6 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class GalaxyDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):

        # Add classes. We have only one class to add.
        self.add_class("dataset", 1, "1_1")
        self.add_class("dataset", 2, "1_2")
        self.add_class("dataset", 3, "1_3")
        self.add_class("dataset", 4, "2_2")
        self.add_class("dataset", 5, "2_3")
        self.add_class("dataset", 6, "3_3")

        # define data locations for images and annotations
        images_dir = dataset_dir + 'images/'
        annotations_dir = dataset_dir + 'annots/'

        # Iterate through all files in the folder to
        # add class, images and annotaions
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]

            # setting image file
            img_path = images_dir + filename

            # setting annotations file
            ann_path = annotations_dir + image_id + '.xml'

            # adding images and annotations to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):

        # load and parse the file
        tree = ElementTree.parse(filename)
        # get the root of the document
        root = tree.getroot()
        # extract each bounding box
        boxes = list()

        for box in root.findall('.//object'):
            xmin = int(box.find('bndbox//xmin').text)
            ymin = int(box.find('bndbox//ymin').text)
            xmax = int(box.find('bndbox//xmax').text)
            ymax = int(box.find('bndbox//ymax').text)
            name = box.find('name').text
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)

        # extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # load the masks for an image
    """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
     """

    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]

        # define anntation  file location
        path = info['annotation']

        # load XML
        boxes, w, h = self.extract_boxes(path)

        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(box[4]))
        return masks, asarray(class_ids, dtype='int32')
        # load an image reference

    """Return the path of the image."""

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        print(info)
        return info['path']


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()

    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id,
                                                                                  use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = modellib.mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"],
                                       r['masks'])
        print(image_id)
        print(gt_class_id)
        print(r["class_ids"])
        # print('image_id = %d class_id = %d detected_class_id = %d' % (image_id, gt_class_id, r["class_ids"]))
        # store
        APs.append(AP)

    # calculate the mean AP across all images
    mAP = math.mean(APs)
    return mAP


if __name__ == '__main__':
    # get command to decide between train and test
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--weights', required=False,
                        default='mask_rcnn_.1577911080.033216.h5',
                        metavar="<weights>",
                        help='.h5 weights file to test ')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Weights: ", args.weights)

    # configure network
    config = myMaskRCNNConfig()
    config.display()

    # prepare train set
    train_set = GalaxyDataset()
    train_set.load_dataset('D1_train/', is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    # prepare test/val set
    test_set = GalaxyDataset()
    test_set.load_dataset('D1_test/', is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    if args.command == "train":
        print("Loading Mask R-CNN model...")
        model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')

        # load the weights
        model.load_weights('../../mask_rcnn_coco.h5',
                           by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        ## train heads with higher lr to speedup the learning
        # model.train(train_set, test_set, learning_rate=2*config.LEARNING_RATE, epochs=1, layers='heads')

        # horizontal flip
        #augmentation = imgaug.augmenters.Flipud(0.5)
        #augmentation = imgaug.augmenters.Noop

        # Training - Stage 1
        print("Training network heads")
        model.train(train_set, test_set,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')
            # ,
            #         augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(train_set, test_set,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')
            # ,
            #         augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(train_set, test_set,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')
            # ,
            #         augmentation=augmentation)

        history = model.keras_model.history.history

        # save final weights
        import time
        model_path = 'mask_rcnn_' + '.' + str(time.time()) + '.h5'
        model.keras_model.save_weights(model_path)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
        model.load_weights(args.weights[1:-1], by_name=True)

        # # evaluate model on training dataset
        # train_mAP = evaluate_model(train_set, model, config)
        # print("Train mAP: %.3f" % train_mAP)
        # # evaluate model on test dataset
        # test_mAP = evaluate_model(test_set, model, config)
        # print("Test mAP: %.3f" % test_mAP)

    if 0:
        from keras.preprocessing.image import load_img
        from keras.preprocessing.image import img_to_array

        # Loading the model in the inference mode
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')
        # loading the trained weights o the custom dataset
        model.load_weights(model_path, by_name=True)
        img = load_img("D4_test/images/FIRSTJ111721.0+342714_infraredctmask.png")
        img = img_to_array(img)
        # detecting objects in the image
        result = model.detect([img])
        print(result)

    if 1:
        image_id = 30
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(test_set, config, image_id,
                                                                                  use_mini_mask=False)
        info = test_set.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               test_set.image_reference(image_id)))
        # Run object detection
        results = model.detect([image], verbose=1)
        # Display results

        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    test_set.class_names, r['scores'],
                                    title="Predictions")
