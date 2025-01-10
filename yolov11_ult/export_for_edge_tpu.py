from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n_trained.pt")  # Load an official model or custom model

# Export the model
model.export(int8=True, format="tflite", data='datasets/aerial_cars_merged.v1i.yolov11/data.yaml')
