import argparse
import cv2
import numpy as np
import os
import tensorflow as tf
import glob


# Funkcja do wczytywania modelu TFLite
def load_model(model_path, use_edgetpu=False):
    if use_edgetpu:
        # Załaduj model z wykorzystaniem Edge TPU
        from tflite_runtime.interpreter import Interpreter, load_delegate
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        # Załaduj standardowy model bez Edge TPU
        interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    return interpreter


# Przygotowanie obrazu wejściowego
def preprocess_image(image_path, input_shape):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, input_shape)  # Zmiana rozmiaru
    input_data = np.expand_dims(image_resized.astype(np.float32), axis=0)  # Dodanie wymiaru batcha
    return input_data / 255.0, image  # Normalizacja do zakresu [0, 1] oraz oryginalny obraz


# Funkcja do stosowania Non-Maximum Suppression (NMS)
def apply_nms(boxes, confidences, nms_threshold=0.4):
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    boxes_nms = np.vstack((x1, y1, x2, y2)).T.astype(np.float32)
    confidences = confidences.astype(np.float32)

    indices = cv2.dnn.NMSBoxes(boxes_nms.tolist(), confidences.tolist(), score_threshold=0.0,
                               nms_threshold=nms_threshold)

    if len(indices) > 0:
        indices = indices.flatten()  # Spłaszczamy, aby usunąć zbędne wymiary
        boxes = boxes[indices]
        confidences = confidences[indices]

    return boxes, confidences


# Przetwarzanie wyników
def process_output(output_data, confidence_threshold, nms_threshold=0.4):
    predictions = output_data[0]  # Rozpakowanie batcha
    scores = predictions[4, :]  # Pewność predykcji
    valid_indices = scores > confidence_threshold  # Filtracja na podstawie pewności
    boxes = predictions[:4, valid_indices].T  # Pobranie współrzędnych dla ważnych predykcji
    confidences = scores[valid_indices]  # Pewności ważnych predykcji

    # Zastosowanie NMS
    boxes, confidences = apply_nms(boxes, confidences, nms_threshold)

    return boxes, confidences


# Rysowanie wykryć na obrazie
def draw_detections(image, boxes, confidences):
    height, width, _ = image.shape
    for box, confidence in zip(boxes, confidences):
        x_center, y_center, w, h = box

        # Obliczamy współrzędne prostokąta
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        # Kolor (możesz zmienić kolor, np. na czerwony)
        color = (0, 255, 0)  # Zielony

        # Rysujemy prostokąt
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Dodajemy tekst z prawdopodobieństwem
        label = f"({confidence:.2f})"
        cv2.putText(image, label, (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


# Główna funkcja
def main():
    # Przetwarzanie argumentów wejściowych
    parser = argparse.ArgumentParser(description="Detekcja obiektów w obrazach przy użyciu modelu TFLite.")
    parser.add_argument('--model', required=True, help="Ścieżka do modelu TFLite")
    parser.add_argument('--imagedir', help="Folder z obrazami do detekcji")
    parser.add_argument('--outputdir', required=True, help="Folder do zapisu wyników detekcji")
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Minimalny próg pewności")
    parser.add_argument('--nms_threshold', type=float, default=0.4, help="Próg NMS")
    parser.add_argument('--draw_image', action='store_true', help="Rysuj detekcje na obrazie (domyślnie wyłączone)")
    parser.add_argument('--edgetpu', action='store_true', help="Użyj akceleratora Edge TPU (domyślnie wyłączone)")

    args = parser.parse_args()

    # Załaduj model
    interpreter = load_model(args.model, use_edgetpu=args.edgetpu)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Przygotowanie obrazu wejściowego
    input_shape = input_details[0]['shape'][1:3]  # Oczekiwany rozmiar wejścia (np. 640x640)

    # Przygotowanie folderów
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # Jeśli podano folder z obrazami, wykonaj detekcje na wszystkich obrazach
    if args.imagedir:
        image_paths = glob.glob(os.path.join(args.imagedir, '*.*'))
    else:
        image_paths = []

    print(f"Found {len(image_paths)} images")

    for image_path in image_paths:
        # Przygotowanie obrazu
        input_data, original_image = preprocess_image(image_path, input_shape)

        # Ustawienie danych wejściowych w interpreterze
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Uruchomienie inferencji
        interpreter.invoke()

        # Pobranie wyników
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Przetworzenie wyników
        boxes, confidences = process_output(output_data, args.confidence_threshold, args.nms_threshold)

        print(f"Found {len(boxes)} objects")

        # Sortowanie wyników po confidence malejąco
        sorted_indices = np.argsort(confidences)[::-1]  # Indeksy posortowane po confidence (malejąco)
        boxes = boxes[sorted_indices]
        confidences = confidences[sorted_indices]

        # Rysowanie detekcji na obrazie (jeśli flaga jest włączona)
        if args.draw_image:
            image_with_detections = draw_detections(original_image, boxes, confidences)
            cv2.imshow("Detekcje", image_with_detections)
            cv2.waitKey(0)  # Czeka na naciśnięcie klawisza

        # Zapisz wyniki detekcji (pozycje i pewność) w pliku tekstowym
        result_txt_path = os.path.join(args.outputdir, os.path.basename(image_path) + '.txt')
        with open(result_txt_path, 'w') as f:
            for box, confidence in zip(boxes, confidences):
                f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {confidence:.6f}\n")

    cv2.destroyAllWindows()

    print("Results saved to", args.outputdir)


# Uruchomienie programu
if __name__ == "__main__":
    main()
