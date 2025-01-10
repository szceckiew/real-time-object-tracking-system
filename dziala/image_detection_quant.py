import argparse
import cv2
import numpy as np
import os
import tensorflow as tf
import glob


# Funkcja do wczytywania modelu TFLite
def load_model(model_path, use_edgetpu=False):
    if use_edgetpu:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    return interpreter


# Funkcja do przygotowania obrazu wejściowego
def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")
    image_resized = cv2.resize(image, input_shape)
    input_data = np.expand_dims(image_resized.astype(np.float32), axis=0)
    return input_data / 255.0, image  # Zwracamy także oryginalny obraz


# Funkcja do przetwarzania wyników detekcji
def process_output(output_data, confidence_threshold, nms_threshold=0.4):
    predictions = output_data[0]  # Rozpakowanie batcha
    confidences = predictions[4, :]  # Pewności predykcji
    valid_indices = confidences > confidence_threshold  # Filtracja na podstawie pewności
    boxes = predictions[:4, valid_indices].T  # Pobranie współrzędnych dla ważnych predykcji
    confidences = confidences[valid_indices]  # Pewności ważnych predykcji

    if len(boxes) > 0:
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_nms = np.vstack((x1, y1, x2, y2)).T
        indices = cv2.dnn.NMSBoxes(boxes_nms.tolist(), confidences.tolist(), confidence_threshold, nms_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = boxes[indices]
            confidences = confidences[indices]

    return boxes, confidences


# Funkcja do rysowania wykryć na obrazie
def draw_detections(image, boxes, confidences):
    height, width, _ = image.shape
    for box, confidence in zip(boxes, confidences):
        x_center, y_center, w, h = box
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        color = (0, 255, 0)  # Zielony
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"({confidence:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


# Główna funkcja
def main():
    parser = argparse.ArgumentParser(description="Detekcja obiektów w obrazach przy użyciu modelu TFLite.")
    parser.add_argument('--model', required=True, help="Ścieżka do modelu TFLite")
    parser.add_argument('--imagedir', required=True, help="Folder z obrazami do detekcji")
    parser.add_argument('--outputdir', required=True, help="Folder do zapisu wyników detekcji")
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Minimalny próg pewności")
    parser.add_argument('--nms_threshold', type=float, default=0.4, help="Próg NMS")
    parser.add_argument('--draw_image', action='store_true', help="Rysuj detekcje na obrazie i zapisz je")
    parser.add_argument('--edgetpu', action='store_true', help="Użyj akceleratora Edge TPU")

    args = parser.parse_args()

    # Załaduj model
    interpreter = load_model(args.model, use_edgetpu=args.edgetpu)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:3]

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    image_paths = glob.glob(os.path.join(args.imagedir, '*.*'))
    print(f"Znaleziono {len(image_paths)} obrazów do przetworzenia.")

    for image_path in image_paths:
        try:
            input_data, original_image = preprocess_image(image_path, input_shape)
        except ValueError as e:
            print(e)
            continue

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        boxes, confidences = process_output(output_data, args.confidence_threshold, args.nms_threshold)

        # Zapis wyników do pliku
        result_txt_path = os.path.join(args.outputdir, f"{os.path.basename(image_path)}.txt")
        with open(result_txt_path, 'w') as f:
            for box, confidence in zip(boxes, confidences):
                f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {confidence:.6f}\n")

        if args.draw_image:
            image_with_detections = draw_detections(original_image, boxes, confidences)
            output_image_path = os.path.join(args.outputdir, f"{os.path.basename(image_path)}_detections.jpg")
            cv2.imwrite(output_image_path, image_with_detections)

    print("Przetwarzanie zakończone. Wyniki zapisano w:", args.outputdir)


if __name__ == "__main__":
    main()
