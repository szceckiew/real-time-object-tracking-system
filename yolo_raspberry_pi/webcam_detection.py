# import argparse
# import cv2
# import numpy as np
# import os
# import tensorflow as tf
# import glob
# import time


# # Funkcja do wczytywania modelu TFLite
# def load_model(model_path, use_edgetpu=False):
#     if use_edgetpu:
#         # Załaduj model z wykorzystaniem Edge TPU
#         from tflite_runtime.interpreter import Interpreter, load_delegate
#         interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
#     else:
#         # Załaduj standardowy model bez Edge TPU
#         interpreter = tf.lite.Interpreter(model_path=model_path)

#     interpreter.allocate_tensors()
#     return interpreter


# # Przygotowanie obrazu wejściowego
# def preprocess_image(image, input_shape):
#     image_resized = cv2.resize(image, input_shape)  # Zmiana rozmiaru
#     input_data = np.expand_dims(image_resized.astype(np.float32), axis=0)  # Dodanie wymiaru batcha
#     return input_data / 255.0  # Normalizacja do zakresu [0, 1]


# # Funkcja do stosowania Non-Maximum Suppression (NMS)
# def apply_nms(boxes, confidences, nms_threshold=0.4):
#     x1 = boxes[:, 0] - boxes[:, 2] / 2
#     y1 = boxes[:, 1] - boxes[:, 3] / 2
#     x2 = boxes[:, 0] + boxes[:, 2] / 2
#     y2 = boxes[:, 1] + boxes[:, 3] / 2

#     boxes_nms = np.vstack((x1, y1, x2, y2)).T.astype(np.float32)
#     confidences = confidences.astype(np.float32)

#     indices = cv2.dnn.NMSBoxes(boxes_nms.tolist(), confidences.tolist(), score_threshold=0.0,
#                                nms_threshold=nms_threshold)

#     if len(indices) > 0:
#         indices = indices.flatten()  # Spłaszczamy, aby usunąć zbędne wymiary
#         boxes = boxes[indices]
#         confidences = confidences[indices]

#     return boxes, confidences


# # Przetwarzanie wyników
# def process_output(output_data, confidence_threshold, nms_threshold=0.4):
#     predictions = output_data[0]  # Rozpakowanie batcha
#     scores = predictions[4, :]  # Pewność predykcji
#     valid_indices = scores > confidence_threshold  # Filtracja na podstawie pewności
#     boxes = predictions[:4, valid_indices].T  # Pobranie współrzędnych dla ważnych predykcji
#     confidences = scores[valid_indices]  # Pewności ważnych predykcji

#     # Zastosowanie NMS
#     boxes, confidences = apply_nms(boxes, confidences, nms_threshold)

#     return boxes, confidences


# # Rysowanie wykryć na obrazie
# def draw_detections(image, boxes, confidences):
#     height, width, _ = image.shape
#     for box, confidence in zip(boxes, confidences):
#         x_center, y_center, w, h = box

#         # Obliczamy współrzędne prostokąta
#         x1 = int((x_center - w / 2) * width)
#         y1 = int((y_center - h / 2) * height)
#         x2 = int((x_center + w / 2) * width)
#         y2 = int((y_center + h / 2) * height)

#         # Kolor (możesz zmienić kolor, np. na czerwony)
#         color = (0, 255, 0)  # Zielony

#         # Rysujemy prostokąt
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

#         # Dodajemy tekst z prawdopodobieństwem
#         label = f"({confidence:.2f})"
#         cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     return image


# # Główna funkcja
# def main():
#     # Przetwarzanie argumentów wejściowych
#     parser = argparse.ArgumentParser(description="Detekcja obiektów w strumieniu wideo przy użyciu modelu TFLite.")
#     parser.add_argument('--model', required=True, help="Ścieżka do modelu TFLite")
#     parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Minimalny próg pewności")
#     parser.add_argument('--nms_threshold', type=float, default=0.4, help="Próg NMS")
#     parser.add_argument('--draw_image', action='store_true', help="Rysuj detekcje na obrazie (domyślnie wyłączone)")
#     parser.add_argument('--save_results', action='store_true', help="Zapisz wyniki do pliku (domyślnie wyłączone)")
#     parser.add_argument('--edgetpu', action='store_true', help="Użyj akceleratora Edge TPU (domyślnie wyłączone)")

#     # Folder do zapisu wyników detekcji tylko jeśli flaga save_results jest włączona
#     parser.add_argument('--outputdir', help="Folder do zapisu wyników detekcji (wymagane, jeśli zapisujesz wyniki)")

#     args = parser.parse_args()

#     # Jeśli zapis wyników jest włączony, sprawdź, czy podano outputdir
#     if args.save_results and not args.outputdir:
#         print("Błąd: Należy podać folder (--outputdir), jeśli zapisywanie wyników jest włączone.")
#         return

#     # Załaduj model
#     interpreter = load_model(args.model, use_edgetpu=args.edgetpu)
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     # Przygotowanie kamery
#     cap = cv2.VideoCapture(0)  # Użyj domyślnej kamery (zmień numer, jeśli używasz innej kamery)

#     if not cap.isOpened():
#         print("Błąd: Nie można otworzyć kamery.")
#         return

#     # Przygotowanie folderu na wyniki, jeśli zapisywanie jest włączone
#     if args.save_results and not os.path.exists(args.outputdir):
#         os.makedirs(args.outputdir)

#     while True:
#         # time.sleep(1)
#         # Pobranie kolejnej klatki z kamery
#         ret, frame = cap.read()
#         if not ret:
#             print("Błąd: Nie udało się pobrać klatki.")
#             break

#         # Przygotowanie obrazu wejściowego
#         input_data = preprocess_image(frame, input_details[0]['shape'][1:3])

#         # Ustawienie danych wejściowych w interpreterze
#         interpreter.set_tensor(input_details[0]['index'], input_data)

#         # Uruchomienie inferencji
#         interpreter.invoke()

#         # Pobranie wyników
#         output_data = interpreter.get_tensor(output_details[0]['index'])

#         # Przetworzenie wyników
#         boxes, confidences = process_output(output_data, args.confidence_threshold, args.nms_threshold)

#         # Sortowanie wyników po confidence malejąco
#         sorted_indices = np.argsort(confidences)[::-1]  # Indeksy posortowane po confidence (malejąco)
#         boxes = boxes[sorted_indices]
#         confidences = confidences[sorted_indices]

#         # Rysowanie detekcji na obrazie (jeśli flaga jest włączona)
#         if args.draw_image:
#             frame = draw_detections(frame, boxes, confidences)

#         # Wyświetlenie obrazu z detekcjami
#         cv2.imshow("Detekcje", frame)

#         # Zapisanie wyników do pliku, jeśli flaga save_results jest włączona
#         if args.save_results:
#             result_txt_path = os.path.join(args.outputdir, 'detekcje.txt')
#             with open(result_txt_path, 'a') as f:
#                 for box, confidence in zip(boxes, confidences):
#                     f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {confidence:.6f}\n")

#         # Jeśli użytkownik naciśnie 'q', zakończ działanie programu
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     print("Zakończono detekcję.")


# # Uruchomienie programu
# if __name__ == "__main__":
#     main()

import argparse
import cv2
import numpy as np
import os
import tensorflow as tf
import time


# Funkcja do wczytywania modelu TFLite
def load_model(model_path, use_edgetpu=False):
    if use_edgetpu:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    return interpreter


# Przygotowanie obrazu wejściowego
def preprocess_image(image, input_shape):
    image_resized = cv2.resize(image, input_shape)  # Zmiana rozmiaru
    input_data = np.expand_dims(image_resized.astype(np.float32), axis=0)  # Dodanie wymiaru batcha
    return input_data / 255.0  # Normalizacja do zakresu [0, 1]


# Przetwarzanie wyników bez NMS
def process_output(output_data, confidence_threshold):
    predictions = output_data[0]  # Rozpakowanie batcha
    scores = predictions[4, :]  # Pewność predykcji
    valid_indices = scores > confidence_threshold  # Filtracja na podstawie pewności
    boxes = predictions[:4, valid_indices]  # Pobranie współrzędnych dla ważnych predykcji
    confidences = scores[valid_indices]  # Pewności ważnych predykcji
    return boxes, confidences


# Rysowanie wykryć na obrazie
def draw_detections(image, boxes, confidences):
    height, width, _ = image.shape
    for box, confidence in zip(boxes.T, confidences):  # Iteracja po kolumnach zamiast transpozycji
        x_center, y_center, w, h = box

        # Obliczamy współrzędne prostokąta
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
    parser = argparse.ArgumentParser(description="Detekcja obiektów w strumieniu wideo przy użyciu modelu TFLite.")
    parser.add_argument('--model', required=True, help="Ścieżka do modelu TFLite")
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Minimalny próg pewności")
    parser.add_argument('--draw_image', action='store_true', help="Rysuj detekcje na obrazie (domyślnie wyłączone)")
    parser.add_argument('--save_results', action='store_true', help="Zapisz wyniki do pliku (domyślnie wyłączone)")
    parser.add_argument('--edgetpu', action='store_true', help="Użyj akceleratora Edge TPU (domyślnie wyłączone)")
    parser.add_argument('--outputdir', help="Folder do zapisu wyników detekcji (wymagane, jeśli zapisujesz wyniki)")

    args = parser.parse_args()

    if args.save_results and not args.outputdir:
        print("Błąd: Należy podać folder (--outputdir), jeśli zapisywanie wyników jest włączone.")
        return

    interpreter = load_model(args.model, use_edgetpu=args.edgetpu)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery.")
        return

    if args.save_results and not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    # Zmienna do obliczania FPS
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie udało się pobrać klatki.")
            break

        # Obliczanie FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Przygotowanie obrazu wejściowego
        input_data = preprocess_image(frame, input_details[0]['shape'][1:3])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        boxes, confidences = process_output(output_data, args.confidence_threshold)
        sorted_indices = np.argsort(confidences)[::-1]
        boxes = boxes[:, sorted_indices]
        confidences = confidences[sorted_indices]

        if args.draw_image:
            frame = draw_detections(frame, boxes, confidences)

        # Wyświetlanie liczby FPS na obrazie
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Detekcje", frame)

        if args.save_results:
            result_txt_path = os.path.join(args.outputdir, 'detekcje.txt')
            with open(result_txt_path, 'a') as f:
                for box, confidence in zip(boxes.T, confidences):  # Iteracja po kolumnach
                    f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {confidence:.6f}\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Zakończono detekcję.")


if __name__ == "__main__":
    main()
