# Convert exported graph file into TFLite model file
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('D:/intelliJ/pycharm_projects/mobilenet/models/custom_model_lite_fpnlite320/saved_model')
tflite_model = converter.convert()

with open('D:/intelliJ/pycharm_projects/mobilenet/models/custom_model_lite_fpnlite320/saved_model/detect.tflite', 'wb') as f:
  f.write(tflite_model)