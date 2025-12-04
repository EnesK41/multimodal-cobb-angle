"""
Model performance evaluation script.
Calculates MAE (Mean Absolute Error) between ground truth and model predictions.
"""
import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp

from config import IMAGES_DIR, MASKS_DIR, MODEL_PATH, DEVICE, ENCODER_NAME, IMG_SIZE
from geometry import calculate_cobb_angle_from_mask


def evaluate_model():
    """Evaluates model performance and calculates MAE."""
    print(f"Calculating MAE... Device: {DEVICE}")

    model = smp.Unet(
        encoder_name=ENCODER_NAME, 
        in_channels=3, 
        classes=1, 
        activation="sigmoid"
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_files = os.listdir(IMAGES_DIR)
    
    results = []
    total_error = 0
    valid_count = 0

    print(f"Total {len(all_files)} files to analyze.")

    for filename in tqdm(all_files):
        img_path = os.path.join(IMAGES_DIR, filename)
        mask_path = os.path.join(MASKS_DIR, filename)

        true_mask_img = cv2.imread(mask_path, 0)
        true_angle, _ = calculate_cobb_angle_from_mask(true_mask_img)
        
        if true_angle == 0.0:
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        x = img_resized.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_mask = model(x)
            pred_mask = pred_mask.cpu().numpy()[0, 0]
        
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
        pred_angle, _ = calculate_cobb_angle_from_mask(pred_mask_binary)

        if pred_angle == 0.0:
            error = true_angle
        else:
            error = abs(true_angle - pred_angle)

        total_error += error
        valid_count += 1
        
        results.append({
            "File": filename,
            "True Angle": round(true_angle, 2),
            "Predicted Angle": round(pred_angle, 2),
            "Error": round(error, 2)
        })

    if valid_count > 0:
        mae = total_error / valid_count
        print(f"\n{'='*40}")
        print(f"Test: {valid_count} images")
        print(f"MAE: {mae:.2f}°")
        print(f"{'='*40}")
        
        df = pd.DataFrame(results)
        df.to_csv("cobb_angle_results.csv", index=False)
        print("Details saved to 'cobb_angle_results.csv'.")
        
        if mae < 5:
            print("Excellent: Below clinical standards!")
        elif mae < 8:
            print("Good: Within acceptable limits.")
        else:
            print("Needs improvement: Error margin is high.")
    else:
        print("Error: Could not calculate angle from any file.")


if __name__ == "__main__":
    evaluate_model()