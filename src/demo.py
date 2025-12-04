"""
Demo script - Runs Cobb angle prediction on a single image.

Usage:
    python demo.py                           # Use first available image
    python demo.py --image path/to/image.jpg # Use specific image
"""
import os
import sys
import glob
import random
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from config import INPUT_DIR, IMAGES_DIR, MODEL_PATH, DEVICE, ENCODER_NAME, IMG_SIZE
from geometry import calculate_cobb_angle_from_mask


def load_model():
    """Loads the trained segmentation model."""
    model = smp.Unet(
        encoder_name=ENCODER_NAME, 
        in_channels=3, 
        classes=1, 
        activation="sigmoid"
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        sys.exit(1)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_mask(model, image):
    """Predicts mask from image."""
    img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    x = img_resized.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_mask = model(x)
        pred_mask = pred_mask.cpu().numpy()[0, 0]
    
    return (pred_mask > 0.5).astype(np.uint8) * 255, img_resized


def visualize_result(img_resized, mask, angle, curve_data, filename):
    """Visualizes the results."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.imshow(mask, alpha=0.4, cmap="jet")
    plt.title(f"Prediction: {filename}")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    
    if curve_data:
        poly_func, y_vals = curve_data
        x_vals = poly_func(y_vals)
        plt.plot(x_vals, y_vals, color="red", linewidth=3, label="Spine Curve")
        plt.text(
            50, 50, f"Cobb: {angle:.1f}°", 
            color="yellow", fontsize=16, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.7)
        )
        plt.legend()
    
    plt.title("Geometric Analysis")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_demo(image_path=None):
    """Runs the demo."""
    if image_path is None:
        # First check input folder
        input_files = (
            glob.glob(os.path.join(INPUT_DIR, "*.png")) +
            glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
            glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
        )
        
        if input_files:
            image_path = input_files[0]
            print(f"Using image from input folder...")
        else:
            # Fall back to augmented dataset with random selection
            augmented_files = (
                glob.glob(os.path.join(IMAGES_DIR, "*.png")) +
                glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
                glob.glob(os.path.join(IMAGES_DIR, "*.jpeg"))
            )
            if not augmented_files:
                print(f"Error: No images found in {INPUT_DIR} or {IMAGES_DIR}")
                return
            image_path = random.choice(augmented_files)
            print(f"Input folder empty, using random image from augmented dataset...")
    
    filename = os.path.basename(image_path)
    print(f"Image: {filename}")
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not read image: {image_path}")
        return
    
    image = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    print("Loading model...")
    model = load_model()
    
    print("Generating mask...")
    mask, img_resized = predict_mask(model, image)
    
    print("Calculating angle...")
    angle, curve_data = calculate_cobb_angle_from_mask(mask)
    
    print(f"Result: Cobb Angle = {angle:.2f}°")
    visualize_result(img_resized, mask, angle, curve_data, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cobb angle demo")
    parser.add_argument("--image", type=str, help="Path to the image to analyze")
    args = parser.parse_args()
    
    run_demo(args.image)