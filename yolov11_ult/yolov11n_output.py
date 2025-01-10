import argparse
import os
import cv2
from ultralytics import YOLO


# Funkcja formatująca wyjście do podanego formatu
def format_detection_output(results, confidence_threshold=0.5):
    formatted_output = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)  # Klasa obiektu
            conf = box.conf.item()  # Pewność detekcji
            if conf >= confidence_threshold:  # Filtruj wg progu pewności
                # Normalizowane współrzędne
                x_center = box.xywh[0][0].item() / result.orig_shape[1]  # Środek X w procentach szerokości
                y_center = box.xywh[0][1].item() / result.orig_shape[0]  # Środek Y w procentach wysokości
                width = box.xywh[0][2].item() / result.orig_shape[1]    # Szerokość w procentach szerokości
                height = box.xywh[0][3].item() / result.orig_shape[0]   # Wysokość w procentach wysokości
                formatted_output.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.2f}")
    return formatted_output


# Funkcja rysująca detekcje na obrazie
def draw_detections(image, results, confidence_threshold=0.5):
    for result in results:
        for box in result.boxes:
            conf = box.conf.item()
            if conf >= confidence_threshold:
                # Współrzędne prostokąta
                x_center = int(box.xywh[0][0].item() * image.shape[1])
                y_center = int(box.xywh[0][1].item() * image.shape[0])
                width = int(box.xywh[0][2].item() * image.shape[1])
                height = int(box.xywh[0][3].item() * image.shape[0])

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                # Rysowanie prostokąta
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


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

    # Utwórz folder wyjściowy, jeśli nie istnieje
    os.makedirs(args.output_folder, exist_ok=True)

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
            image_with_detections = draw_detections(frame, results, confidence_threshold=args.confidence)

            # Wyświetlenie obrazu z detekcjami
            cv2.imshow("Video Detections", image_with_detections)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Naciśnij 'q', aby przerwać
                break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
