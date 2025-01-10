import torch
from ultralytics import YOLO

# Załaduj model
model = YOLO("yolo11n_trained.pt")

# Wymiary wejściowe modelu (domyślnie 640x640, dostosuj jeśli inny rozmiar był użyty podczas treningu)
input_shape = (1, 3, 640, 640)  # batch_size=1, channels=3, height=640, width=640
dummy_input = torch.randn(*input_shape)

# Testowanie wejścia
output = model.model(dummy_input)  # Przekazanie danych do modelu bazowego (PyTorch)

# Sprawdź, co zawiera output
print("Output:", output)

# Jeśli output jest krotką, sprawdź jej zawartość
if isinstance(output, tuple):
    for i, out in enumerate(output):
        print(f"Output {i} shape:", out.shape if hasattr(out, 'shape') else "No shape attribute")
