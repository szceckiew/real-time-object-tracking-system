import os
import requests
import tarfile

# Zdefiniowanie wybranego modelu i konfiguracji
chosen_model = 'ssd-mobilenet-v2-fpnlite-320'

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

# Stworzenie folderu na model
base_dir = 'D:/intelliJ/pycharm_projects/mobilenet/models/mymodel_ssd-mobilenet-v2-fpnlite-320/'
os.makedirs(base_dir, exist_ok=True)
os.chdir(base_dir)

# Pobranie pretrenowanego modelu
download_tar = f'http://download.tensorflow.org/models/object_detection/tf2/20200711/{pretrained_checkpoint}'
print(f"Pobieranie: {download_tar}")

response = requests.get(download_tar, stream=True)
with open(pretrained_checkpoint, 'wb') as file:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            file.write(chunk)

# Rozpakowanie pliku .tar.gz
with tarfile.open(pretrained_checkpoint, 'r:gz') as tar:
    tar.extractall()

print("Model rozpakowany.")

# Pobranie pliku konfiguracyjnego
download_config = f'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/{base_pipeline_file}'
print(f"Pobieranie: {download_config}")

config_response = requests.get(download_config)
config_path = os.path.join(base_dir, base_pipeline_file)
with open(config_path, 'wb') as file:
    file.write(config_response.content)

print(f"Konfiguracja zapisana do {config_path}.")
