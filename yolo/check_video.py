import cv2
import os


def read_detections(detection_file):
    """Wczytuje detekcje z pliku. Plik zawiera dane w formacie: x_center, y_center, w, h"""
    detections = []
    with open(detection_file, 'r') as file:
        for line in file:
            x_center, y_center, w, h = map(float, line.strip().split())
            detections.append((x_center, y_center, w, h))
    return detections


def draw_detections(image, detections, frame_number):
    """Rysuje detekcje na obrazie i numer ramki"""
    height, width = image.shape[:2]


    # Rysowanie detekcji
    for detection in detections:
        x_center, y_center, w, h = detection

        # Obliczamy współrzędne prostokąta
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        color = (0, 255, 0)

        print(x1, y1, x2, y2)

        # Rysujemy prostokąt
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Dodajemy tekst z prawdopodobieństwem poniżej prostokąta
        label = f"Object"
        cv2.putText(image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Dodajemy pierwszą współrzędną x_center poniżej prostokąta
        x_center_text = f"x: {x_center:.5f}"
        cv2.putText(image, x_center_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Wyświetlamy numer ramki w lewym górnym rogu
    frame_text = f"Frame: {frame_number:06d}"
    cv2.putText(image, frame_text, (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image


def main():
    video_path = 'cars_moving.mp4'  # Ścieżka do pliku wideo
    detections_folder = 'model_outputs/frames-1lost_moving'  # Ścieżka do folderu z detekcjami

    # Otwórz wideo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Nie udało się otworzyć pliku wideo.")
        return

    # Parametry wideo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_number = 0
    while True:
        # Odczytujemy klatkę wideo
        ret, frame = cap.read()
        if not ret:
            print("Koniec wideo.")
            break

        # Tworzymy nazwę pliku detekcji z odpowiednim formatowaniem
        detection_file = os.path.join(detections_folder, f'frame_{frame_number:06d}.txt')

        # Jeśli plik detekcji istnieje, wczytujemy detekcje
        if os.path.exists(detection_file):
            detections = read_detections(detection_file)
            frame = draw_detections(frame, detections, frame_number)

        # Wyświetlamy zmodyfikowaną klatkę
        cv2.imshow('Frame with Detections', frame)

        # Czekamy na naciśnięcie klawisza
        key = cv2.waitKey(0) & 0xFF  # Oczekujemy na naciśnięcie dowolnego klawisza

        if key == ord('q'):  # Jeśli naciśnięto 'q', kończymy program
            break
        elif key == ord(' '):  # Jeśli naciśnięto spację, przechodzimy do kolejnej klatki
            frame_number += 1
            continue

    # Zakończenie pracy
    cap.release()
    cv2.destroyAllWindows()
    print("Zakończono wyświetlanie wideo.")


if __name__ == '__main__':
    main()
