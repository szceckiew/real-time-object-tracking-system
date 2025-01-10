import argparse
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def convert_bbox_format(bbox):
    x_center, y_center, width, height = bbox
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return [x1, y1, x2, y2]

def calculate_iou(pred_bbox, gt_bbox):
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox
    x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox

    xi1 = max(x1_pred, x1_gt)
    yi1 = max(y1_pred, y1_gt)
    xi2 = min(x2_pred, x2_gt)
    yi2 = min(y2_pred, y2_gt)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    union = pred_area + gt_area - intersection
    iou = intersection / union if union > 0 else 0
    return iou

def calculate_precision_recall(detections, ground_truth, iou_threshold=0.5):
    if not detections:
        return 0, 0
    if not ground_truth:
        return 1, 0

    true_positives = 0
    false_positives = 0
    matched_gt = set()

    for detection in detections:
        pred_bbox = convert_bbox_format(detection[1:5])
        match_found = False

        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            gt_bbox = convert_bbox_format(gt[1:5])
            iou = calculate_iou(pred_bbox, gt_bbox)

            if iou >= iou_threshold:
                match_found = True
                matched_gt.add(i)
                break

        if match_found:
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ground_truth) - len(matched_gt)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    return precision, recall

def calculate_ap_from_detections(all_detections, all_ground_truth, iou_threshold=0.5):
    all_detections = sorted(all_detections, key=lambda x: x[5], reverse=True)  # Sort by confidence
    tp_cumulative = 0
    fp_cumulative = 0
    total_gt = len(all_ground_truth)

    precisions = []
    recalls = []

    matched_gt = set()
    for detection in all_detections:
        pred_bbox = convert_bbox_format(detection[1:5])
        match_found = False

        for i, gt in enumerate(all_ground_truth):
            if i in matched_gt:
                continue
            gt_bbox = convert_bbox_format(gt[1:5])
            iou = calculate_iou(pred_bbox, gt_bbox)
            if iou >= iou_threshold:
                match_found = True
                matched_gt.add(i)
                break

        if match_found:
            tp_cumulative += 1
        else:
            fp_cumulative += 1

        precision = tp_cumulative / (tp_cumulative + fp_cumulative)
        recall = tp_cumulative / total_gt if total_gt > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    ap = 0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap

def load_results_from_file(file_path):
    detections = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            class_id = int(values[0])
            bbox = list(map(float, values[1:5]))
            confidence = float(values[5])
            detections.append([class_id] + bbox + [confidence])
    return detections

def load_ground_truth_from_file(file_path):
    ground_truth = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            class_id = int(values[0])
            bbox = list(map(float, values[1:5]))
            ground_truth.append([class_id] + bbox)
    return ground_truth

def evaluate_model_results(detection_files, ground_truth_files, iou_threshold=0.5):
    global_detections = []
    global_ground_truth = []
    results = []

    for detection_file, ground_truth_file in zip(detection_files, ground_truth_files):
        detections = load_results_from_file(detection_file)
        ground_truth = load_ground_truth_from_file(ground_truth_file)
        precision, recall = calculate_precision_recall(detections, ground_truth, iou_threshold)
        results.append((os.path.basename(detection_file), precision, recall))

        global_detections.extend(detections)
        global_ground_truth.extend(ground_truth)

    ap = calculate_ap_from_detections(global_detections, global_ground_truth, iou_threshold)
    return results, ap

def get_files_from_directory(directory, extension=".txt"):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)])

def main():
    parser = argparse.ArgumentParser(description="Evaluate detection results against ground truth.")
    parser.add_argument("detections_folder", help="Path to the folder containing detection results.")
    parser.add_argument("ground_truth_folder", help="Path to the folder containing ground truth annotations.")
    parser.add_argument("-o", "--output_file", type=str, help="Path to save the evaluation results.")
    parser.add_argument("-t", "--iou_threshold", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    args = parser.parse_args()

    detection_files = get_files_from_directory(args.detections_folder)
    ground_truth_files = get_files_from_directory(args.ground_truth_folder)

    if len(detection_files) != len(ground_truth_files):
        logging.warning("The number of detection files does not match the number of ground truth files.")

    results, ap = evaluate_model_results(detection_files, ground_truth_files, args.iou_threshold)

    if args.output_file:
        with open(args.output_file, "w") as file:
            for detection_file, precision, recall in results:
                file.write(f"{detection_file} - Precision: {precision:.4f}, Recall: {recall:.4f}\n")
            file.write(f"Average Precision (AP): {ap:.4f}\n")
        logging.info(f"Results saved to {args.output_file}")

    logging.info(f"Average Precision (AP): {ap:.4f}")

if __name__ == "__main__":
    main()
