from ultralytics import YOLO
import onnx

# Załaduj model
model = YOLO("yolo11n_trained.pt")  # Zmień ścieżkę, jeśli to konieczne

# Eksport do ONNX
model.export(format="onnx")
