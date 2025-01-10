import cv2
import os
import argparse


def draw_detections(image, detections):
    height, width = image.shape[:2]

    for detection in detections:
        class_id, x_center, y_center, w, h, prob = detection

        # Obliczamy współrzędne prostokąta
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        color = (0, 255, 0)

        # Rysujemy prostokąt
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Dodajemy tekst z prawdopodobieństwem
        label = f"Class {class_id} ({prob:.2f})"
        cv2.putText(image, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def load_detections_from_file(file_path):
    detections = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            class_id = int(values[0])
            x_center, y_center, w, h = map(float, values[1:5])
            probability = float(values[5])
            detections.append([class_id, x_center, y_center, w, h, probability])
    return detections


def get_files_from_directory(directory, extension=".txt"):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)])


def main():
    parser = argparse.ArgumentParser(description="Rysowanie detekcji na obrazach.")
    parser.add_argument("images_folder", help="Ścieżka do folderu z obrazkami.")
    parser.add_argument("detections_folder", help="Ścieżka do folderu z plikami detekcji.")
    parser.add_argument("--output_folder", help="Ścieżka do folderu, gdzie mają być zapisane obrazy z detekcjami.",
                        default=None)
    args = parser.parse_args()

    image_files = get_files_from_directory(args.images_folder, ".png")  # Zmieniamy na .jpg, ale można też na .png
    detection_files = get_files_from_directory(args.detections_folder, ".txt")

    if len(image_files) != len(detection_files):
        print("Ostrzeżenie: Liczba plików z obrazkami nie pasuje do liczby plików z detekcjami.")

    for image_file, detection_file in zip(image_files, detection_files):
        # Wczytujemy obraz
        image = cv2.imread(image_file)

        # Wczytujemy detekcje
        detections = load_detections_from_file(detection_file)

        # Rysowanie detekcji na obrazie
        image_with_detections = draw_detections(image, detections)

        # Wyświetlenie obrazu z detekcjami
        cv2.imshow(f"Detekcje - {os.path.basename(image_file)}", image_with_detections)

        # Jeśli podano folder wyjściowy, zapisujemy obraz
        if args.output_folder:
            # Sprawdzamy, czy folder wyjściowy istnieje, jeśli nie, to go tworzymy
            if not os.path.exists(args.output_folder):
                os.makedirs(args.output_folder)

            # Zapisujemy obraz
            output_image_path = os.path.join(args.output_folder,
                                             os.path.splitext(os.path.basename(image_file))[0] + "_detected_yolo_edge.jpg")
            cv2.imwrite(output_image_path, image_with_detections)
            print(f"Zapisano obraz z detekcjami: {output_image_path}")

        # Czekanie na naciśnięcie klawisza, aby przejść do następnego obrazu
        cv2.waitKey(0)

    # Zamknięcie wszystkich okien
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
