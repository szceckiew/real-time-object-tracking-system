import cv2
import numpy as np
import os
from sort.sort import *

# Funkcja do wczytywania danych z plików
def load_detections(file_path):
    detections = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            # Przekształcamy dane na odpowiedni format: [x_center, y_center, w, h]
            x_center, y_center, w, h = map(float, parts)

            width = 1280
            height = 720

            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)

            detections.append([x1, y1, x2, y2, 1])
    return np.array(detections)

# Funkcja do przetwarzania wszystkich klatek i przeprowadzenia śledzenia
def track_objects(detection_folder, video_path, output_folder, max_files=118, max_frames=118):
    tracker = Sort()  # Inicjalizacja algorytmu SORT
    tracked_objects = []
    all_centers = {}  # Zmienna do przechowywania wszystkich środków obiektów

    # Otwórz wideo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Nie udało się otworzyć pliku wideo.")
        return []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Tworzymy folder na zapisane klatki, jeśli nie istnieje

    last_frame = None  # Zmienna do przechowywania ostatniej klatki przetworzonej

    # Przechodzimy po wszystkich plikach z wykryciami (jednym na klatkę) i ograniczamy do max_files
    detection_files = sorted(os.listdir(detection_folder))[:max_files]

    # Zmienna do liczenia przetworzonych klatek
    frame_count = 0

    for filename in detection_files:
        file_path = os.path.join(detection_folder, filename)

        if os.path.isfile(file_path) and filename.endswith(".txt"):
            detections = load_detections(file_path)

            # Śledzenie obiektów
            trackers = tracker.update(detections)

            # Wczytaj bieżącą klatkę wideo
            ret, frame = cap.read()
            if not ret:
                break

            # Zwiększamy licznik klatek
            frame_count += 1

            # Rysowanie bounding boxów i ID na klatce
            for obj in trackers:
                x1, y1, x2, y2, obj_id = obj

                # Obliczanie środków obiektów
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Dodajemy punkt do listy środków obiektów
                if obj_id not in all_centers:
                    all_centers[obj_id] = []
                all_centers[obj_id].append((center_x, center_y))

                # Rysowanie prostokąta na klatce
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Pozycja tekstu (pod bbox)
                text_position = (int(x1), int(y2) + 20)  # Przesuwamy w dół od dolnej krawędzi bboxa
                # Dodanie tekstu z ID obiektu pod prostokątem
                cv2.putText(frame, f"ID: {int(obj_id)}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Przechowaj ostatnią klatkę
            last_frame = frame.copy()

            # Ścieżka do zapisu obrazu
            image_filename = f"{filename.split('.')[0]}.png"
            output_path = os.path.join(output_folder, image_filename)

            # Zapisz klatkę z naniesionymi informacjami
            cv2.imwrite(output_path, frame)

            # Wyświetlanie klatki z naniesionymi informacjami (opcjonalne)
            cv2.imshow(f"Klatka {filename}", frame)

            key = cv2.waitKey(1) & 0xFF
            # Czekanie na naciśnięcie klawisza, aby przejść do następnej klatki
            if key == ord('q'):
                break
            elif key == ord(' '):  # Jeśli naciśnięto spację, przechodzimy do kolejnej klatki
                continue

            # Przerwij, jeśli osiągnięto limit klatek
            if frame_count >= max_frames:
                break

    # Rysowanie linii łączących środki obiektów na ostatniej klatce
    if last_frame is not None:
        # Rysowanie linii łączących środki obiektów
        for obj_id, centers in all_centers.items():
            for i in range(1, len(centers)):
                start_point = (int(centers[i-1][0]), int(centers[i-1][1]))
                end_point = (int(centers[i][0]), int(centers[i][1]))

                # Rysowanie linii
                cv2.line(last_frame, start_point, end_point, (0, 0, 255), 2)  # Czerwony kolor dla linii

        # Zapisanie obrazu z liniami
        final_image_path = os.path.join(output_folder, "final_with_lines.png")
        cv2.imwrite(final_image_path, last_frame)

        # Wyświetlanie finalnej klatki
        cv2.imshow("Finalna klatka z liniami", last_frame)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# Ścieżka do folderu z plikami z wykryciami, wideo oraz folderu na zapisane obrazy
detection_folder = "detections/frames-20lost"
video_path = "cars_moving.mp4"  # Zamień na ścieżkę do swojego pliku wideo
output_folder = "output_images"  # Folder na zapisane obrazy

# Śledzenie obiektów, zapisanie wyników na wideo i zapisanie obrazów
track_objects(detection_folder, video_path, output_folder)
