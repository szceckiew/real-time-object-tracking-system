import tensorflow as tf

# Wczytanie modelu
converter = tf.lite.TFLiteConverter.from_saved_model('path_to_saved_model')

# Ustawienia kwantyzacji
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.representative_dataset = representative_data_gen

# Konwersja modelu
quantized_model = converter.convert()

# Zapisanie skwantyzowanego modelu
with open('model_quantized.tflite', 'wb') as f:
    f.write(quantized_model)
