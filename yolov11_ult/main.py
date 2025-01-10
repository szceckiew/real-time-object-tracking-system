import os
import argparse
from ultralytics import YOLO


# Function to remove the labels.cache file
def remove_cache_file(cache_file):
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f'File {cache_file} was removed.')
    else:
        print(f'File {cache_file} does not exist.')


# Main function
def main(args):
    # Path to the dataset folder
    dataset_dir = args.dataset_dir  # Path to the folder containing the dataset

    # Create the full path to the data.yaml file
    data_file = os.path.join(dataset_dir, "data.yaml")

    # Create the full path to the labels.cache file
    cache_file = os.path.join(dataset_dir, "valid", "labels.cache")

    # If the --remove-cache flag is set, remove the labels.cache file
    if args.remove_cache:
        remove_cache_file(cache_file)

    # Load the specified model, or default to yolov8n.pt if no model is provided
    model_path = args.model if args.model else "yolov11n.pt"
    model = YOLO(model_path)  # Load the model specified by the argument or default to yolov8n.pt

    # Run evaluation
    metrics = model.val(data=data_file, split="val")  # split="val" means the validation dataset

    # Get the Average Precision (AP) and Average Precision at IoU=0.5 (AP50) metrics
    ap = metrics.results_dict.get('metrics/mAP50-95(B)', 'mAP50-95 metric does not exist anymore')  # mAP for IoU range 0.5-0.95
    ap50 = metrics.results_dict.get('metrics/mAP50(B)', 'mAP50 metric does not exist anymore')  # mAP for IoU=0.5

    # Display results
    print(f'Average Precision (AP): {ap}')
    print(f'Average Precision (AP50): {ap50}')

    # Save the results to a text file
    output_file = "average_precision_results.txt"  # Path to the file where the results will be saved

    with open(output_file, "w") as file:
        file.write(f'Average Precision (AP) for this model: {ap}\n')
        file.write(f'Average Precision (AP50) for this model: {ap50}\n')

    print(f'AP and AP50 were saved to: {output_file}')


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Yolov8 Evaluation script")
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help="Path to dataset directory (standard Yolov8 type, only validation folder required)"
    )
    parser.add_argument(
        '--remove_cache',
        action='store_true',
        help="Remove saved cache file"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Path to the model file. If not provided, 'yolov11n.pt' will be used as default."
    )

    return parser.parse_args()


# Main program part
if __name__ == "__main__":
    args = parse_args()  # Parse arguments
    main(args)  # Run the main function
