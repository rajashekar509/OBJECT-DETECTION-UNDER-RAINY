from utils.evaluation import calculate_map
import os

def evaluate_performance(ground_truth_folder, detection_results_folder):
    """
    Evaluate object detection performance using mAP.
    """
    # Calculate mAP
    map_score = calculate_map(ground_truth_folder, detection_results_folder)
    print(f"mAP: {map_score:.4f}")

if __name__ == "__main__":
    # Evaluate performance
    ground_truth_folder = 'data/clear_images'  # Folder with ground truth annotations
    detection_results_folder = 'data/derained_images'  # Folder with detection results
    evaluate_performance(ground_truth_folder, detection_results_folder)