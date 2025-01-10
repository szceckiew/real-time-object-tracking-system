import onnx
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from onnx_tf.backend import prepare
from tensorflow.lite import TFLiteConverter

# Załaduj model ONNX
onnx_model = onnx.load("yolov8n.onnx")

# Konwersja modelu ONNX na TensorFlow
onnx_model_tf = prepare(onnx_model)

# Przekształć TensorFlow model do formy odpowiedniej do konwersji TFLite
tf_model = onnx_model_tf.tf_module

# Możesz wykonać konwersję na stałe wartości (jeśli model wymaga tego)
full_model = convert_variables_to_constants_v2(tf_model)

# Skonfiguruj konwerter TFLite
converter = TFLiteConverter.from_concrete_functions([full_model.signatures['serving_default']])

# Wybór opcji optymalizacji kwantyzacji
converter.optimizations = [TFLiteConverter.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # Określenie typu wejścia dla kwantyzacji
converter.inference_output_type = tf.uint8  # Określenie typu wyjścia

# Możliwość przeprowadzenia kwantyzacji
def representative_dataset_gen():
    # Zwróć przykładowe dane wejściowe (np. obrazki)
    for _ in range(100):
        yield [tf.random.normal([1, 640, 640, 3], dtype=tf.float32)]

converter.representative_dataset = representative_dataset_gen

# Konwertuj model
tflite_model = converter.convert()

# Zapisz model TFLite do pliku
with open("yolov8n_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("Model został przekonwertowany na TFLite i zapisany.")
