import os
model_path = 'edgetpu.tflite'
model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"Rozmiar: {model_size:.2f} MB")