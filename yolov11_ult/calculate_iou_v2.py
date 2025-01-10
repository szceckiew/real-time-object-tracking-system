import os
import glob
import numpy as np
import argparse


def yolo_to_bbox(txt_data, image_width=1, image_height=1):
    """Konwertuje format YOLO do formatu [xmin, ymin, xmax, ymax]."""
    boxes = []
    for line in txt_data:
        parts = line.strip().split()
        if len(parts) < 5:  # Pomijamy niepełne linie
            continue
        _, x_center, y_center, width, height, *confidence = map(float, parts)
        xmin = (x_center - width / 2) * image_width
        ymin = (y_center - height / 2) * image_height
        xmax = (x_center + width / 2) * image_width
        ymax = (y_center + height / 2) * image_height
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def calculate_centroid(box):
    """Oblicza centroid bounding boxa."""
    x_min, y_min, x_max, y_max = box
    return [(x_min + x_max) / 2, (y_min + y_max) / 2]


def calculate_iou(box1, box2):
    """Oblicza IoU dla dwóch bounding boxów."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def match_and_evaluate(ground_truth, detections, distance_threshold=50, iou_threshold=0.5):
    """
    Dopasowuje bounding boxy na podstawie minimalnej odległości centroidów
    i oblicza IoU dla dopasowanych par.
    """
    matched = []
    unmatched_detections = set(range(len(detections)))
    unmatched_ground_truth = set(range(len(ground_truth)))
    iou_scores = []

    gt_centroids = [calculate_centroid(gt) for gt in ground_truth]
    det_centroids = [calculate_centroid(det) for det in detections]

    for gt_idx, gt_centroid in enumerate(gt_centroids):
        best_distance = float('inf')
        best_det_idx = None
        for det_idx in unmatched_detections:
            distance = np.linalg.norm(np.array(gt_centroid) - np.array(det_centroids[det_idx]))
            if distance < best_distance and distance <= distance_threshold:
                best_distance = distance
                best_det_idx = det_idx
        if best_det_idx is not None:
            iou = calculate_iou(ground_truth[gt_idx], detections[best_det_idx])
            if iou >= iou_threshold:
                matched.append((gt_idx, best_det_idx, iou))
                iou_scores.append(iou)
                unmatched_detections.remove(best_det_idx)
                unmatched_ground_truth.remove(gt_idx)

    false_positives = len(unmatched_detections)
    false_negatives = len(unmatched_ground_truth)
    mean_iou = np.mean(iou_scores) if iou_scores else 0

    return {
        "matched_pairs": matched,
        "mean_iou": mean_iou,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "num_pairs": len(matched),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate bounding boxes from ground truth and detection TXT files.")
    parser.add_argument("ground_truth", type=str, help="Path to the folder with ground truth TXT files.")
    parser.add_argument("detections", type=str, help="Path to the folder with detection TXT files.")
    parser.add_argument("--distance_threshold", type=float, default=50, help="Centroid distance threshold.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results.")
    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.ground_truth, "*.txt")))
    det_files = sorted(glob.glob(os.path.join(args.detections, "*.txt")))

    if not gt_files or not det_files:
        print("Error: No files found in one or both input folders.")
        return

    overall_results = []
    total_mean_iou_numerator = 0  # Licznik dla średniej IoU ważonej liczbą par
    total_matched_pairs = 0       # Łączna liczba dopasowanych par
    total_false_positives = 0     # Łączna liczba false positives
    total_false_negatives = 0     # Łączna liczba false negatives

    for gt_file, det_file in zip(gt_files, det_files):
        with open(gt_file, 'r') as f:
            gt_data = f.readlines()
        with open(det_file, 'r') as f:
            det_data = f.readlines()

        ground_truth = yolo_to_bbox(gt_data)
        detections = yolo_to_bbox(det_data)

        results = match_and_evaluate(
            ground_truth=ground_truth,
            detections=detections,
            distance_threshold=args.distance_threshold,
            iou_threshold=args.iou_threshold,
        )
        overall_results.append({
            "file": os.path.basename(gt_file),
            "results": results,
        })

        # Aktualizacja dla podsumowania wyników
        total_mean_iou_numerator += results["mean_iou"] * results["num_pairs"]
        total_matched_pairs += results["num_pairs"]
        total_false_positives += results["false_positives"]
        total_false_negatives += results["false_negatives"]

    # Obliczenie końcowej średniej IoU ważonej
    overall_mean_iou = total_mean_iou_numerator / total_matched_pairs if total_matched_pairs > 0 else 0

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            for result in overall_results:
                f.write(f"File: {result['file']}\n")
                f.write(f"Matched Pairs: {result['results']['matched_pairs']}\n")
                f.write(f"Mean IoU: {result['results']['mean_iou']:.4f}\n")
                f.write(f"False Positives: {result['results']['false_positives']}\n")
                f.write(f"False Negatives: {result['results']['false_negatives']}\n")
                f.write(f"Number of Matched Pairs: {result['results']['num_pairs']}\n\n")
            f.write(f"Overall Mean IoU (weighted): {overall_mean_iou:.4f}\n")
            f.write(f"Total False Positives: {total_false_positives}\n")
            f.write(f"Total False Negatives: {total_false_negatives}\n")
        print(f"Results saved to {args.output}")
    else:
        for result in overall_results:
            print(f"File: {result['file']}")
            print(f"Matched Pairs: {result['results']['matched_pairs']}")
            print(f"Mean IoU: {result['results']['mean_iou']:.4f}")
            print(f"False Positives: {result['results']['false_positives']}")
            print(f"False Negatives: {result['results']['false_negatives']}")
            print(f"Number of Matched Pairs: {result['results']['num_pairs']}\n")
        print(f"Overall Mean IoU (weighted): {overall_mean_iou:.4f}")
        print(f"Total False Positives: {total_false_positives}")
        print(f"Total False Negatives: {total_false_negatives}")


if __name__ == "__main__":
    main()
