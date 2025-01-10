import tensorflow as tf

# Ustawić widoczność wszystkich GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s).")
    tf.config.set_visible_devices(physical_devices, 'GPU')
else:
    print("No GPU found.")

print("TF CUDA available: ", tf.test.is_built_with_cuda())
print("TF GPU available: ", tf.test.is_gpu_available())
