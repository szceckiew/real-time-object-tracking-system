# Set training parameters for the model
num_steps = 5000

train_record_fname = 'D:/intelliJ/pycharm_projects/mobilenet/datasets/aerial_cars_merged.v1i.tfrecord/train/train.tfrecord'
val_record_fname = 'D:/intelliJ/pycharm_projects/mobilenet/datasets/aerial_cars_merged.v1i.tfrecord/valid/val.tfrecord'
label_map_pbtxt_fname = 'D:/intelliJ/pycharm_projects/mobilenet/datasets/aerial_cars_merged.v1i.tfrecord/train/labelmap.pbtxt'

mymodel_path = "mymodel_ssd-mobilenet-v2-fpnlite-320"
chosen_model = 'ssd-mobilenet-v2-fpnlite-320'
training_folder = "training_ssd-mobilenet-v2-fpnlite-320"

MODELS_CONFIG = {
    'ssd-mobilenet-v2': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
    },
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
    },
    'ssd-mobilenet-v2-fpnlite-320': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
    },
}

model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']

if chosen_model == 'efficientdet-d0':
    batch_size = 4
else:
    batch_size = 16

# Set file locations and get number of classes for config file
pipeline_fname = f'D:/intelliJ/pycharm_projects/mobilenet/models/{mymodel_path}/{base_pipeline_file}'
fine_tune_checkpoint = f'D:/intelliJ/pycharm_projects/mobilenet/models/{mymodel_path}/{model_name}/checkpoint/ckpt-0'


def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())


num_classes = get_num_classes(label_map_pbtxt_fname)
print('Total classes:', num_classes)

# Create custom configuration file by writing the dataset, model checkpoint, and training parameters into the base pipeline file
import re
print('Writing custom configuration file')

with open(pipeline_fname) as f:
    s = f.read()
with open(f'D:/intelliJ/pycharm_projects/mobilenet/models/{mymodel_path}/pipeline_file.config', 'w') as f:

    # Set fine_tune_checkpoint path
    s = re.sub('fine_tune_checkpoint: ".*?"',
               f'fine_tune_checkpoint: "{fine_tune_checkpoint}"', s)

    # Set tfrecord files for train and test datasets
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', f'input_path: "{train_record_fname}"', s)
    s = re.sub(
        '(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', f'input_path: "{val_record_fname}"', s)

    # Set label_map_path
    s = re.sub(
        'label_map_path: ".*?"', f'label_map_path: "{label_map_pbtxt_fname}"', s)

    # Set batch_size
    s = re.sub('batch_size: [0-9]+',
               f'batch_size: {batch_size}', s)

    # Set training steps, num_steps
    s = re.sub('num_steps: [0-9]+',
               f'num_steps: {num_steps}', s)

    # Set number of classes num_classes
    s = re.sub('num_classes: [0-9]+',
               f'num_classes: {num_classes}', s)

    # Change fine-tune checkpoint type from "classification" to "detection"
    s = re.sub(
        'fine_tune_checkpoint_type: "classification"', f'fine_tune_checkpoint_type: "detection"', s)

    # If using ssd-mobilenet-v2, reduce learning rate (because it's too high in the default config file)
    if chosen_model == 'ssd-mobilenet-v2':
        s = re.sub('learning_rate_base: .8',
                   'learning_rate_base: .08', s)

        s = re.sub('warmup_learning_rate: 0.13333',
                   'warmup_learning_rate: .026666', s)

    # If using efficientdet-d0, use fixed_shape_resizer instead of keep_aspect_ratio_resizer (because it isn't supported by TFLite)
    if chosen_model == 'efficientdet-d0':
        s = re.sub('keep_aspect_ratio_resizer', 'fixed_shape_resizer', s)
        s = re.sub('pad_to_max_dimension: true', '', s)
        s = re.sub('min_dimension', 'height', s)
        s = re.sub('max_dimension', 'width', s)

    f.write(s)

# (Optional) Display the custom configuration file's contents
with open(f'D:/intelliJ/pycharm_projects/mobilenet/models/{mymodel_path}/pipeline_file.config', 'r') as f:
    print(f.read())

# Set the path to the custom config file and the directory to store training checkpoints in
pipeline_file = f'D:/intelliJ/pycharm_projects/mobilenet/models/{mymodel_path}/pipeline_file.config'
model_dir = f'D:/intelliJ/pycharm_projects/mobilenet/{training_folder}/'

# Run training!
import os
os.system(f'python D:/intelliJ/pycharm_projects/mobilenet/models/research/object_detection/model_main_tf2.py '
          f'--pipeline_config_path={pipeline_file} '
          f'--model_dir={model_dir} '
          f'--alsologtostderr '
          f'--num_train_steps={num_steps} '
          f'--sample_1_of_n_eval_examples=1')
