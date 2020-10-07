#!/usr/bin/env python3

# base imports
import numpy as np
import os
import shutil
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
print(tf.__version__)
import time
from datetime import datetime
import pandas as pd

import zipfile
import six

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# Import the object detection module
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

################################################################################
#                              Model preparation
################################################################################

# Model loader
def load_model(model_name):
  directory = 'models/research/object_detection/test_data/' #CHECK!
  model_dir = directory + model_name + "/saved_model"
  if os.path.exists(model_dir):
    model = tf.saved_model.load(model_dir)
    return model
  else:
    # exits the program
    sys.exit('model name incorrectly spelt or model not loaded')

# Loading label map-List of the used strings to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt' #CHECK!
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# path to test images
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/shared-folder/input_folder') #CHECK!
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

# path to save tested images and object counts
PATH_TO_SAVE_IMAGES_DIR = pathlib.Path('/shared-folder/output_folder') #CHECK!

################################################################################
#                                  Detection
################################################################################

# Load an object detection model
# chose a model from models/research/object_detection/test_data/
model_name = 'efficientdet_d1_coco17_tpu-32'
detection_model = load_model(model_name)

# Add a wrapper function to call the model, and cleanup the outputs
def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
      
  return output_dict

# Define and retrieve attributes of objects identified in images
def get_detections(image_name, image, boxes, classes, scores, cat_index, min_score_thresh):
    im_width, im_height = image.size
    detections = []
    for i in range(boxes.shape[0]):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            (left, right, top, bottom) = (xmin * im_width,
                                          xmax * im_width,
                                          ymin * im_height,
                                          ymax * im_height)
            if classes[i] in cat_index.keys():
                class_name = cat_index[classes[i]]['name']
            else:
                class_name='N/A'
            detections.append(
                {'image_id': image_name,
                 'object': class_name,
                 'coordinates': {
                     'left': left,
                     'right': right,
                     'bottom': bottom,
                     'top': top
                 },
                 'score': scores[i]
                 }
            )
    return detections


# Run inference on each test image and show the results

# Set min_score_thresh to other values (between 0 and 1) to allow more 
# detections in or to filter out more detections.
overall_detections = [] 
def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  threshold = 0.5 # set minimun score threshold
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=threshold,
      line_thickness=8)
  
  image_name = os.path.basename(image_path)
  image = Image.open(image_path)

  detections = get_detections(
      image_name,
      image,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      threshold)
  
  overall_detections.extend(detections)      
  
  # save images with bounding boxes
  im_save = Image.fromarray(image_np)
  image_name = os.path.basename(image_path)
  im_save.save(PATH_TO_SAVE_IMAGES_DIR + '/' + image_name) #add '_bb' for bounding boxes?


# Run inference for all images in TEST_IMAGE_PATHS directory and plot 
# average time per image
elapsed = [] #CHECK!
for image_path in TEST_IMAGE_PATHS:
  start_time = time.time() #CHECK!
  show_inference(detection_model, image_path)
  end_time = time.time() #CHECK!
  elapsed.append(end_time - start_time) #CHECK!
  # move tested images from input folder
  shutil.move(image_path, '/shared-folder/archive_folder')#no need to include the file name in destination  #CHECK!

mean_elapsed = sum(elapsed) / float(len(elapsed)) #CHECK!
print('Elapsed time: ' + str(mean_elapsed) + ' second per image') #CHECK!



################################################################################
#                          Count objects of interest
################################################################################

# Dataframe with image detections from pretrained model
df = pd.DataFrame(overall_detections)
print('overall detections:') #CHECK!
print(df.head()) #CHECK!
df = df.iloc[:,1:]

# group by image and type of object and perform counts
objects_of_interest = ['bicycle', 'car', 'person', 'motorcycle', 'bus', 'truck']
df = df[df.object.isin(objects_of_interest)]
df = df[['image_id', 'object']]
df1 = df.groupby(['image_id', 'object']).size().to_frame('counts').reset_index()

# transpose table with objects as columns
df1 = df1.pivot_table(index='image_id', columns='object', values='counts', fill_value=0)

# add absent columns with zero value
absent_objects = [obj for obj in objects_of_interest if obj not in df1.columns]

if absent_objects:
    for obj in absent_objects:
        df1[obj] = 0

df1['timestamp'] = datetime.now() #Return the current local date and time
df1 = df1.reset_index()
# reorder columns
df1 = df1[['timestamp','image_id','car','person','bicycle','motorcycle','bus','truck']]
df1.columns.name = None

# append dataframe to csv report
if os.path.isfile(PATH_TO_SAVE_IMAGES_DIR + '/report.csv'):
    df1.to_csv(PATH_TO_SAVE_IMAGES_DIR + '/report.csv', index=False, mode='a', header=False)
else:
    df1.rename(columns={'image_id':'image'}, inplace=True)
    df1.to_csv(PATH_TO_SAVE_IMAGES_DIR + '/report.csv', index=False)
