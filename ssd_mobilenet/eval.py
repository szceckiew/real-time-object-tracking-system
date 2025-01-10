from object_detection import model_lib_v2

mymodel_path = "mymodel_ssd-mobilenet-v2-fpnlite-320"
training_folder_path = "training_ssd-mobilenet-v2-fpnlite-320"

model_lib_v2.eval_continuously(
    pipeline_config_path=f"D:/intelliJ/pycharm_projects/mobilenet/models/{mymodel_path}/pipeline_file.config",
    model_dir=f"D:/intelliJ/pycharm_projects/mobilenet/{training_folder_path}",
    checkpoint_dir=f"D:/intelliJ/pycharm_projects/mobilenet/{training_folder_path}",
    eval_timeout=3600  # Maksymalny czas (w sekundach) na znalezienie nowych checkpointów
)