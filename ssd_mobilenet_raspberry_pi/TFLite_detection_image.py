######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras (Modified by ChatGPT)
# Date: 11/11/22
# Modified: [Today's Date]
#
# Description: 
# Modified to allow user to specify the output directory for text results.
#
# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default=None)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
parser.add_argument('--outputdir', help='Directory to save text results. Default is "results".',
                    default='results')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

IM_NAME = args.image
IM_DIR = args.imagedir

# Set output directory
OUTPUT_DIR = args.outputdir

# If both an image AND a folder are specified, throw an error
if (IM_NAME and IM_DIR):
    print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
    sys.exit()

# If neither an image or a folder are specified, default to using 'test1.jpg' for image name
if (not IM_NAME and not IM_DIR):
    IM_NAME = 'test1.jpg'

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Define path to images and grab all image filenames
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)

# Create output directory if not exists
RESULTS_PATH = os.path.join(CWD_PATH, OUTPUT_DIR)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Remove first label if it is '???'
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): 
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: 
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Loop over every image and perform detection
for image_path in images:

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    detections = []

    # Loop over detections
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            ymin = boxes[i][0]
            xmin = boxes[i][1]
            ymax = boxes[i][2]
            xmax = boxes[i][3]

            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width_norm = xmax - xmin
            height_norm = ymax - ymin

            detections.append([labels[int(classes[i])], x_center, y_center, width_norm, height_norm, scores[i]])

    # Save results in a text file only
    image_fn = os.path.basename(image_path)
    base_fn, ext = os.path.splitext(image_fn)
    txt_result_fn = base_fn + '.txt'
    txt_savepath = os.path.join(RESULTS_PATH, txt_result_fn)

    with open(txt_savepath, 'w') as f:
        for detection in detections:
            object_name = detection[0]
            x_center = detection[1]
            y_center = detection[2]
            width_norm = detection[3]
            height_norm = detection[4]
            confidence = detection[5]

            f.write(f"{object_name} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f} {confidence:.4f}\n")

# Clean up
cv2.destroyAllWindows()
