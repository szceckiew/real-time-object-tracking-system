import os
import argparse
from ultralytics import YOLO

# Funkcja do przetwarzania obrazów w folderze
def process_images(input_folder, output_folder, model_path):
    # Ładowanie modelu YOLO
    tflite_model = YOLO(model_path)

    # Sprawdzenie, czy folder wyjściowy istnieje, jeśli nie, to go tworzymy
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterowanie po plikach w folderze wejściowym
    for filename in os.listdir(input_folder):
        # Sprawdzamy, czy plik jest obrazem (można dodać inne rozszerzenia, jeśli potrzeba)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {image_path}...")

            # Uruchamianie detekcji na obrazie
            results = tflite_model(image_path, task="detect")

            # Generowanie ścieżki do pliku wyjściowego
            output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

            with open(output_file, "w") as file:
                for result in results:  # Dla każdego wykrytego obiektu w wynikach
                    data = result.boxes.cpu().numpy()

                    print(data)

                    for detection in range(len(data)):
                        confidence = data.conf[detection]

                        # Obliczanie współrzędnych w formacie x_center, y_center, w, h
                        x_center = data.xywhn[detection][0]
                        y_center = data.xywhn[detection][1]
                        w = data.xywhn[detection][2]
                        h = data.xywhn[detection][3]

                        # Zapisanie do pliku
                        file.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} {confidence:.6f}\n")

            print(f"Results saved to {output_file}")

# Funkcja główna do obsługi argumentów z linii komend
def main():
    # Ustawienia argumentów linii komend
    parser = argparse.ArgumentParser(description="YOLO object detection on multiple images.")
    parser.add_argument("input_folder", help="Path to the folder containing input images.")
    parser.add_argument("output_folder", help="Path to the folder where results will be saved.")
    parser.add_argument("model_path", help="Path to the YOLO model (e.g. model.tflite or model.pt).")
    args = parser.parse_args()

    # Przetwarzanie obrazów w folderze
    process_images(args.input_folder, args.output_folder, args.model_path)

if __name__ == "__main__":
    main()
