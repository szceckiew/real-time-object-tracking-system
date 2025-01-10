output_directory = "D:/intelliJ/pycharm_projects/mobilenet/models/custom_model_lite_fpnlite320"

# Path to training directory (the conversion script automatically chooses the highest checkpoint file)
last_model_path = 'D:/intelliJ/pycharm_projects/mobilenet/training_ssd-mobilenet-v2-fpnlite-320'
pipeline_file = 'D:/intelliJ/pycharm_projects/mobilenet/models/mymodel_ssd-mobilenet-v2-fpnlite-320/pipeline_file.config'

import os
os.system(f'python D:/intelliJ/pycharm_projects/mobilenet/models/research/object_detection/export_tflite_graph_tf2.py '
          f'--trained_checkpoint_dir {last_model_path} '
          f'--output_directory {output_directory} '
          f'--pipeline_config_path {pipeline_file} ')


