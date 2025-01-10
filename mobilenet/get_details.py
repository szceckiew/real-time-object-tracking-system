import tensorflow as tf

def get_info(model_path):
    print(model_path)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("input_details", input_details)
    print("output_details", output_details)

get_info('models/custom_model_lite/detect_quant.tflite')