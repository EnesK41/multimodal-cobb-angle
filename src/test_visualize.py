"""
Test visualization script - Shows model performance on random images.
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from config import IMAGES_DIR, MASKS_DIR, MODEL_PATH, DEVICE, ENCODER_NAME, IMG_SIZE


def visualize_prediction():
    """Visualizes model predictions on randomly selected images."""
    print(f"Loading model: {MODEL_PATH}")
    
    model = smp.Unet(
        encoder_name=ENCODER_NAME, 
        encoder_weights=None,
        in_channels=3, 
        classes=1, 
        activation="sigmoid"
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_files = os.listdir(IMAGES_DIR)
    test_files = np.random.choice(all_files, min(5, len(all_files)), replace=False)

    plt.figure(figsize=(15, 10))
    
    for idx, filename in enumerate(test_files):
        img_path = os.path.join(IMAGES_DIR, filename)
        mask_path = os.path.join(MASKS_DIR, filename)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_mask = cv2.imread(mask_path, 0)
        
        img_input = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_input = img_input.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_mask = model(img_tensor)
            pred_mask = pred_mask.cpu().numpy()[0, 0]
        
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
        
        plt.subplot(5, 3, idx*3 + 1)
        plt.imshow(image)
        plt.title(f"Input: {filename}", fontsize=8)
        plt.axis("off")
        
        plt.subplot(5, 3, idx*3 + 2)
        plt.imshow(true_mask, cmap="gray")
        plt.title("Ground Truth", fontsize=8)
        plt.axis("off")

        plt.subplot(5, 3, idx*3 + 3)
        plt.imshow(pred_mask_binary, cmap="gray")
        plt.title("Prediction", fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("test_results.png")
    print("Results saved to 'test_results.png'.")
    plt.show()


if __name__ == "__main__":
    visualize_prediction()