import argparse
import os
import cv2
import shutil
from ultralytics import YOLO


# Funkcja formatująca wyjście do podanego formatu (bez klas i pewności)
def format_detection_output(results, confidence_threshold=0.5):
    formatted_output = []
    for result in results:
        for box in result.boxes:
            conf = box.conf.item()
            if conf >= confidence_threshold:
                # Normalizowane współrzędne
                x_center = box.xywh[0][0].item() / result.orig_shape[1]  # Środek X w procentach szerokości
                y_center = box.xywh[0][1].item() / result.orig_shape[0]  # Środek Y w procentach wysokości
                width = box.xywh[0][2].item() / result.orig_shape[1]    # Szerokość w procentach szerokości
                height = box.xywh[0][3].item() / result.orig_shape[0]   # Wysokość w procentach wysokości
                formatted_output.append(f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return formatted_output


# Funkcja rysująca detekcje na obrazie
def draw_detections(image, detections):
    # print("detections", detections)
    height, width = image.shape[:2]

    for detection in detections:
        # print("detection", detection)
        detection = detection.strip().split()
        print("detection", detection)

        x_center, y_center, w, h = map(float, detection[:4])

        # Obliczamy współrzędne prostokąta
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        color = (0, 255, 0)

        # Rysujemy prostokąt
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Dodajemy tekst z prawdopodobieństwem
        label = f"Object"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def clear_output_folder(folder_path):
    """Usuwa zawartość folderu wyjściowego."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Usuwa cały folder wraz z zawartością
    os.makedirs(folder_path)  # Tworzy pusty folder


def main():
    # Parsowanie argumentów
    parser = argparse.ArgumentParser(description="YOLOv11n Object Detection on a Video")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("output_folder", help="Path to the folder where detections will be saved")
    parser.add_argument(
        "-m", "--model", default="yolo11n.pt", help="Path to the YOLOv11n model (default: yolov11n.pt)"
    )
    parser.add_argument(
        "-c", "--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "-d", "--display", action="store_true", help="Display video with detections (default: off)"
    )
    args = parser.parse_args()

    # Wczytaj model YOLO
    model = YOLO(args.model)

    # Usuń zawartość folderu wyjściowego
    clear_output_folder(args.output_folder)

    # Otwórz plik wideo
    cap = cv2.VideoCapture(args.video_path)
    frame_id = 0

    if not cap.isOpened():
        print(f"Error: Could not open video {args.video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_id}")

        # Wykonaj predykcję
        results = model(frame)

        # Sformatuj wyjście
        output = format_detection_output(results, confidence_threshold=args.confidence)


        # Zapisz wynik w osobnym pliku w podanym folderze wyjściowym
        output_file = os.path.join(args.output_folder, f"frame_{frame_id:06d}.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(output))

        print(f"Saved detections to: {output_file}")

        # Rysowanie detekcji na obrazie i wyświetlanie (jeśli flaga -d jest ustawiona)
        if args.display:
            image_with_detections = draw_detections(frame, output)

            # Wyświetlenie obrazu z detekcjami
            cv2.imshow("Video Detections", image_with_detections)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Naciśnij 'q', aby przerwać
                break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
