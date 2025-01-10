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
def track_objects(detection_folder, video_path, output_folder):
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

    # Przechodzimy po wszystkich plikach z wykryciami (jednym na klatkę)
    for filename in sorted(os.listdir(detection_folder)):
        file_path = os.path.join(detection_folder, filename)

        if os.path.isfile(file_path) and filename.endswith(".txt"):
            detections = load_detections(file_path).astype(np.float32)  # Konwersja na float32

            # Śledzenie obiektów
            trackers = tracker.update(detections, img_size, img_size)  # Poprawne przekazanie wymiarów

            # Wczytaj bieżącą klatkę wideo
            ret, frame = cap.read()
            if not ret:
                break

            # Rysowanie bounding boxów i ID na klatce
            for obj in trackers:
                track_id = obj.track_id
                tlhw = obj.tlwh

                x1, y1, w, h = [int(coord) for coord in tlhw]
                x2 = x1 + w
                y2 = y1 + h

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

    cap.release()
    cv2.destroyAllWindows()

# Ścieżka do folderu z plikami z wykryciami, wideo oraz folderu na zapisane obrazy
detection_folder = "detections/frames-1lost_moving"
video_path = "cars_moving.mp4"  # Zamień na ścieżkę do swojego pliku wideo
output_folder = "detections/bytetrack_output"  # Folder na zapisane obrazy

# Śledzenie obiektów i zapisanie wyników na wideo oraz zapisanie obrazów
track_objects(detection_folder, video_path, output_folder)
