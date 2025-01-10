import os
import argparse

def append_value_to_lines_in_files(folder_path, value="0.66"):
    # Sprawdź, czy folder istnieje
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} nie istnieje.")
        return

    # Iteruj po wszystkich plikach w folderze
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Sprawdź, czy to plik (pomijamy foldery)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Zapisz zmodyfikowane linie z wartością na końcu
            with open(file_path, 'w') as file:
                for line in lines:
                    # Usuń końcowy znak nowej linii, dodaj wartość, a potem zapisz z powrotem
                    file.write(line.strip() + " " + value + "\n")

            print(f"Zaktualizowano plik: {file_path}")

if __name__ == "__main__":
    # Użycie argparse do obsługi argumentów linii komend
    parser = argparse.ArgumentParser(description="Dodaj wartość do końca każdej linii w plikach w folderze.")
    parser.add_argument("folder_path", help="Ścieżka do folderu z plikami")
    parser.add_argument("--value", default="0.66", help="Wartość do dodania na końcu linii (domyślnie 0.66)")

    args = parser.parse_args()

    # Uruchom funkcję
    append_value_to_lines_in_files(args.folder_path, args.value)
