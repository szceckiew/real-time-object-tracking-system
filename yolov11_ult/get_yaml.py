from ultralytics import YOLO
import yaml  # Biblioteka do pracy z plikami YAML

# Załaduj przetrenowany model
model = YOLO("worse.pt")

# Eksportuj konfigurację modelu do pliku YAML
with open("worse.yaml", "w") as f:
    yaml.dump(model.model.yaml, f, default_flow_style=False, sort_keys=False)
print("Plik konfiguracyjny zapisany jako yolo11n.yaml")
