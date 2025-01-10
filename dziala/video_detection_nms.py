import argparse
import cv2
import numpy as np
import os
import tensorflow as tf
import time

def load_model(model_path, use_edgetpu=False):
    if use_edgetpu:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        interpreter = Interpreter(model_path=model_path, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, input_shape):
    image_resized = cv2.resize(image, input_shape)
    input_data = np.expand_dims(image_resized.astype(np.float32), axis=0)
    return input_data / 255.0

def process_output(output_data, confidence_threshold):
    predictions = output_data[0]
    scores = predictions[4, :]
    valid_indices = scores > confidence_threshold
    boxes = predictions[:4, valid_indices]
    confidences = scores[valid_indices]
    return boxes, confidences

def apply_nms(boxes, confidences, nms_threshold):
    if len(boxes) == 0:
        return np.array([]), np.array([])

    boxes_tlbr = []
    for box in boxes.T:
        x_center, y_center, w, h = box
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        boxes_tlbr.append([x1, y1, x2, y2])

    boxes_tlbr = np.array(boxes_tlbr)
    indices = cv2.dnn.NMSBoxes(boxes_tlbr.tolist(), confidences.tolist(), score_threshold=0, nms_threshold=nms_threshold)

    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[:, indices], confidences[indices]
    else:
        return np.array([]), np.array([])

def draw_detections(image, boxes, confidences):
    height, width, _ = image.shape
    for box, confidence in zip(boxes.T, confidences):
        x_center, y_center, w, h = box

        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = f"({confidence:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def main():
    parser = argparse.ArgumentParser(description="Detekcja obiektów w strumieniu wideo przy użyciu modelu TFLite.")
    parser.add_argument('--model', required=True, help="Ścieżka do modelu TFLite")
    parser.add_argument('--video', required=True, help="Ścieżka do pliku wideo")
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Minimalny próg pewności")
    parser.add_argument('--nms_threshold', type=float, default=0.4, help="Próg NMS")
    parser.add_argument('--draw_image', action='store_true', help="Rysuj detekcje na obrazie (domyślnie wyłączone)")
    parser.add_argument('--save_results', action='store_true', help="Zapisz wyniki do pliku (domyślnie wyłączone)")
    parser.add_argument('--edgetpu', action='store_true', help="Użyj akceleratora Edge TPU (domyślnie wyłączone)")
    parser.add_argument('--outputdir', help="Folder do zapisu wyników detekcji (wymagane, jeśli zapisujesz wyniki)")

    args = parser.parse_args()

    if args.save_results and not args.outputdir:
        print("Błąd: Należy podać folder (--outputdir), jeśli zapisywanie wyników jest włączone.")
        return

    interpreter = load_model(args.model, use_edgetpu=args.edgetpu)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć pliku wideo.")
        return

    if args.save_results and not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie udało się pobrać klatki.")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        input_data = preprocess_image(frame, input_details[0]['shape'][1:3])
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        boxes, confidences = process_output(output_data, args.confidence_threshold)
        boxes, confidences = apply_nms(boxes, confidences, args.nms_threshold)

        if args.draw_image:
            frame = draw_detections(frame, boxes, confidences)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Detekcje", frame)

        if args.save_results:
            result_txt_path = os.path.join(args.outputdir, 'detekcje.txt')
            with open(result_txt_path, 'a') as f:
                for box, confidence in zip(boxes.T, confidences):
                    f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {confidence:.6f}\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Zakończono detekcję.")

if __name__ == "__main__":
    main()
