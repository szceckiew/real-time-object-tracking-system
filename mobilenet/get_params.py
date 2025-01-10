import tensorflow as tf

# Wczytanie modelu z formatu SavedModel
model = tf.saved_model.load('D:/intelliJ/pycharm_projects/mobilenet/models/custom_model_lite/saved_model')

# Model w formacie SavedModel może mieć różne komponenty. Często główny komponent modelu to "serving_default".
# Musisz odwołać się do funkcji w modelu, aby uzyskać dostęp do zmiennych.

# Jeśli model jest prostym modelem, to zmienne można znaleźć w głównym obiekcie
# "model.variables", ale jeśli model jest bardziej złożony, np. posiada różne komponenty,
# trzeba odwołać się do ich zmiennych.

# Możesz sprawdzić dostępne elementy w modelu:
print("Model zawiera następujące komponenty:")
for key in model.signatures:
    print(key)

# Przykład: odwołanie do 'serving_default' (jeśli to jest dostępny klucz)
serving_default = model.signatures['serving_default']

# Zmienna modelu powinna być dostępna w atrybucie variables w kontekście warstwy modelu:
total_params = 0
for var in serving_default.trainable_variables:
    total_params += tf.size(var).numpy()

print(f"Liczba parametrów w modelu: {total_params}")
