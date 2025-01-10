from ultralytics import YOLO


def get_info(model_path):
    # Ładowanie modelu
    model = YOLO(model_path)

    # Wyświetlanie szczegółów modelu
    info = model.info()  # Zwraca szczegóły modelu

    # Wyświetlanie informacji o wejściu i wyjściu
    print(f"Model structure for {model_path}:")
    print(model)  # Wyświetla ogólną strukturę modelu

    # Jeśli model.info() jest słownikiem
    if isinstance(info, dict):
        print(f"Input shape: {info.get('input')}")
        print(f"Output shape: {info.get('output')}")
        print(f"Model details: {info}")
    else:
        # W przypadku, gdy info jest krotką, próbujemy zwrócić odpowiednie dane
        print(f"Model info is a tuple: {info}")


# Wywołanie funkcji z ścieżką do modelu
get_info("yolo11n_trained_integer_quant.tflite")
