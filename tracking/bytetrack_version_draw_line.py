import cv2
import numpy as np
import os
from yolox.tracker.byte_tracker import BYTETracker  # import algorytmu ByteTrack

# Klasa do przekazania argumentów do BYTETracker
class TrackerArgs:
    def __init__(self):
        self.track_thresh = 0.5  # Próg pewności śledzenia
        self.track_buffer = 30  # Maksymalna liczba utraconych klatek
        self.match_thresh = 0.8  # Próg dopasowania
        self.mot20 = False       # Czy używać konfiguracji dla MOT20?

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

            detections.append([x1, y1, x2, y2, 1.0])  # Użyj float dla confidence
    return np.array(detections, dtype=np.float32)  # Konwersja na float32

# Funkcja do przetwarzania wszystkich klatek i przeprowadzenia śledzenia
def track_objects(detection_folder, video_path, output_folder, max_files=118, max_frames=118):
    args = TrackerArgs()  # Utwórz obiekt `args`
    tracker = BYTETracker(args)  # Inicjalizacja ByteTrack z argumentami

    # Otwórz wideo
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Nie udało się otworzyć pliku wideo.")
        return []

    # Pobierz rozmiar wideo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_size = np.array([width, height], dtype=np.float32)  # Wymiary klatki wideo jako float32

    # Upewnij się, że folder do zapisywania wyników istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Tworzymy folder na zapisane klatki, jeśli nie istnieje

    # Przechodzimy po wszystkich plikach z wykryciami (jednym na klatkę) i ograniczamy do max_files
    detection_files = sorted(os.listdir(detection_folder))[:max_files]

    # Zmienna do liczenia przetworzonych klatek
    frame_count = 0

    # Przechowywanie środków obiektów
    all_centers = {}

    # Przechodzimy po plikach wykryć
    for filename in detection_files:
        file_path = os.path.join(detection_folder, filename)

        if os.path.isfile(file_path) and filename.endswith(".txt"):
            detections = load_detections(file_path).astype(np.float32)  # Konwersja na float32

            # Śledzenie obiektów
            trackers = tracker.update(detections, img_size, img_size)  # Poprawne przekazanie wymiarów

            # Wczytaj bieżącą klatkę wideo
            ret, frame = cap.read()
            if not ret:
                break

            # Zwiększamy licznik klatek
            frame_count += 1

            # Rysowanie bounding boxów i ID na klatce
            for obj in trackers:
                track_id = obj.track_id
                tlhw = obj.tlwh

                x1, y1, w, h = [int(coord) for coord in tlhw]
                x2 = x1 + w
                y2 = y1 + h

                # Obliczanie środka obiektu
                center_x = int(x1 + w / 2)
                center_y = int(y1 + h / 2)

                # Przechowywanie środków obiektów
                if track_id not in all_centers:
                    all_centers[track_id] = []
                all_centers[track_id].append((center_x, center_y))

                # Rysowanie prostokąta na klatce
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Dodanie tekstu z ID obiektu (pod bbox)
                cv2.putText(frame, f"ID: {int(track_id)}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Ścieżka do zapisu obrazu
            image_filename = f"{filename.split('.')[0]}.png"
            output_path = os.path.join(output_folder, image_filename)

            # Zapisz klatkę z naniesionymi informacjami
            cv2.imwrite(output_path, frame)

            # Wyświetlanie klatki z naniesionymi informacjami
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

    # Na końcu rysowanie linii łączących środki obiektów na ostatniej klatce
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się wczytać ostatniej klatki.")
        return

    for track_id, centers in all_centers.items():
        # Rysowanie prostokątów i ID dla ostatniej klatki
        for obj in trackers:
            if obj.track_id == track_id:
                tlhw = obj.tlwh
                x1, y1, w, h = [int(coord) for coord in tlhw]
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {int(track_id)}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Rysowanie linii łączącej środki obiektów
        for i in range(1, len(centers)):
            # Sprawdzanie czy nie ma przerwy w wykryciach
            if centers[i-1] and centers[i]:
                start_point = (int(centers[i-1][0]), int(centers[i-1][1]))
                end_point = (int(centers[i][0]), int(centers[i][1]))

                # Rysowanie linii łączącej środki obiektów
                cv2.line(frame, start_point, end_point, (0, 0, 255), 2)  # Czerwony kolor dla linii

    # Zapisanie finalnej klatki z liniami i detekcjami
    final_image_path = os.path.join(output_folder, "final_with_lines_and_detections.png")
    cv2.imwrite(final_image_path, frame)

    # Wyświetlanie finalnej klatki z liniami i detekcjami
    cv2.imshow("Finalna klatka z liniami i detekcjami", frame)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# Ścieżka do folderu z plikami z wykryciami, wideo oraz folderu na zapisane obrazy
detection_folder = "detections/frames-20lost"
video_path = "cars_moving.mp4"  # Zamień na ścieżkę do swojego pliku wideo
output_folder = "detections/bytetrack_output"  # Folder na zapisane obrazy

# Śledzenie obiektów i zapisanie wyników na wideo oraz zapisanie obrazów
track_objects(detection_folder, video_path, output_folder)
