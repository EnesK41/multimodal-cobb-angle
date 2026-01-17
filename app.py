"""
Cobb Angle Measurement System

Automatic source detection for spine Cobb angle measurement.
Supports Hospital DRR, TotalSegmentator DRR, labeled X-ray, and unknown inputs.
"""

import os
import re
import time

import cv2
import gradio as gr
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def detect_source_type(filepath):
    """Detect source type from filepath."""
    if not filepath or filepath.strip() == "":
        print("[detect_source_type] Empty file path, source type: unknown")
        return "unknown", None, None
    filepath_lower = filepath.lower().replace("\\", "/")
    filename = os.path.basename(filepath)
    filename_lower = filename.lower()
    dirname = os.path.dirname(filepath)
    print(f"[detect_source_type] File: {filepath} (filename: {filename})")
    if filename_lower.startswith('sunhl') or 'labeled_xray' in filepath_lower:
        match = re.search(r'_gt(\d+(?:\.\d+)?)', filename, re.IGNORECASE)
        gt_angle = float(match.group(1)) if match else None
        print(f"[detect_source_type] X-ray detected. GT angle: {gt_angle}")
        return "xray", gt_angle, None
    if 'hospital_drr' in filepath_lower:
        mask_path = os.path.join(dirname, '..', 'masks', filename)
        mask_path = os.path.normpath(mask_path)
        print(f"[detect_source_type] Hospital DRR detected. Mask path: {mask_path} (exists: {os.path.exists(mask_path)})")
        return "hospital", None, mask_path if os.path.exists(mask_path) else None
    if 'verse_drr' in filepath_lower or 'totalsegmentator' in filepath_lower:
        mask_path = os.path.join(dirname, '..', 'masks', filename)
        mask_path = os.path.normpath(mask_path)
        print(f"[detect_source_type] TotalSegmentator DRR detected. Mask path: {mask_path} (exists: {os.path.exists(mask_path)})")
        return "totalsegmentator", None, mask_path if os.path.exists(mask_path) else None
    print("[detect_source_type] Source type: unknown")
    return "unknown", None, None


class ResnetBlock(nn.Module):
    """ResNet block for CycleGAN generator."""
    
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )
    
    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """ResNet-based generator for CycleGAN."""
    
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        for i in range(2):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, 2, 1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        mult = 4
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]
        
        for i in range(2):
            mult = 4 // (2 ** i)
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, 2, 1, 1),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True)
            ]
        
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


cyclegan_model = None
unet_model = None


def load_models():
    """Load CycleGAN and U-Net models."""
    global cyclegan_model, unet_model
    
    try:
        print("[load_models] Loading CycleGAN model...")
        cyclegan_model = ResnetGenerator()
        cyclegan_model.load_state_dict(
            torch.load('cyclegan_xray_to_drr.pth', map_location=DEVICE)
        )
        cyclegan_model.to(DEVICE).eval()
        print("[load_models] CycleGAN model loaded.")
        print("[load_models] Loading U-Net model...")
        unet_model = smp.Unet(
            encoder_name='resnet34',
            in_channels=3,
            classes=18,
            activation=None
        )
        unet_model.load_state_dict(
            torch.load('best_spine_unet.pth', map_location=DEVICE)
        )
        unet_model.to(DEVICE).eval()
        print("[load_models] U-Net model loaded.")
        return True
    except Exception as e:
        print(f"[load_models] Model loading error: {e}")
        return False


cyclegan_transform = transforms.Compose([
    transforms.Resize((1024, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

unet_transform = A.Compose([
    A.Resize(1024, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def calculate_cobb_angle(mask, label_offset=0):
    """Calculate Cobb angle using PCA on vertebrae masks.
    
    Args:
        mask: Segmentation mask with vertebrae labels
        label_offset: Offset for label conversion (7 for TotalSegmentator, 0 for others)
    
    Returns:
        tuple: (cobb_angle, num_vertebrae)
    """
    vertebrae_info = []
    label_range = range(8, 25) if label_offset > 0 else range(1, 18)
    
    for label_id in label_range:
        coords = np.argwhere(mask == label_id)
        if len(coords) < 100:
            continue
        
        y_center = coords[:, 0].mean()
        pca = PCA(n_components=1).fit(coords)
        slope = np.degrees(np.arctan2(pca.components_[0][0], pca.components_[0][1]))
        vertebra_id = label_id - label_offset if label_offset > 0 else label_id
        vertebrae_info.append((vertebra_id, y_center, slope, len(coords)))
    
    if len(vertebrae_info) < 2:
        return None, 0
    
    vertebrae_info.sort(key=lambda x: x[3], reverse=True)
    top_vertebrae = vertebrae_info[:min(10, len(vertebrae_info))]
    top_vertebrae.sort(key=lambda x: x[1])
    
    if len(top_vertebrae) < 2:
        return None, 0
    
    slopes = [v[2] for v in top_vertebrae]
    cobb_angle = abs(max(slopes) - min(slopes))
    
    return cobb_angle, len(top_vertebrae)


def process_image(image, filepath):
    """Main image processing function.
    
    Args:
        image: Input image (numpy array or PIL Image)
        filepath: File path for source detection
    
    Returns:
        tuple: (gallery_images, results_markdown)
    """
    if image is None:
        print("[process_image] No image, waiting for upload.")
        return [], "Please upload an image"
    print(f"[process_image] Analysis started. File path: {filepath}")
    start_time = time.time()
    
    source_type, gt_angle_from_filename, gt_mask_path = detect_source_type(filepath)
    print(f"[process_image] Source type: {source_type}, GT mask path: {gt_mask_path}")
    if isinstance(image, Image.Image):
        print("[process_image] PIL Image, converting to numpy.")
        image = np.array(image)
    if len(image.shape) == 2:
        print("[process_image] Grayscale image, converting to RGB.")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        print("[process_image] RGBA image, converting to RGB.")
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = cv2.resize(image, (512, 1024))
    print(f"[process_image] Image resized: {image.shape}")
    original_input = image.copy()
    fake_drr_np = None
    use_cyclegan = source_type in ["xray", "unknown"]
    if use_cyclegan:
        print("[process_image] Generatig DRR with CycleGAN...")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        pil_img = Image.fromarray(gray).convert('RGB')
        cyclegan_input = cyclegan_transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            fake_drr = cyclegan_model(cyclegan_input)
        fake_drr_np = fake_drr.squeeze().cpu().numpy().transpose(1, 2, 0)
        fake_drr_np = ((fake_drr_np * 0.5 + 0.5) * 255).astype(np.uint8)
        process_image_for_unet = fake_drr_np
        print("[process_image] CycleGAN DRR generated.")
    else:
        process_image_for_unet = image.copy()
    if source_type in ["hospital", "totalsegmentator"]:
        print(f"[process_image] Vertical flip applying for {source_type}.")
        process_image_for_unet = cv2.flip(process_image_for_unet, 0)
    print("[process_image] U-Net segmentation starting...")
    unet_input = unet_transform(image=process_image_for_unet)['image'].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        segmentation = unet_model(unet_input)
        pred_mask = torch.argmax(segmentation, dim=1).squeeze().cpu().numpy()
    print("[process_image] U-Net segmentation completed.")
    if source_type in ["hospital", "totalsegmentator"]:
        print(f"[process_image] Re-flipping mask and input for {source_type}.")
        pred_mask = cv2.flip(pred_mask.astype(np.uint8), 0)
        process_image_for_unet = cv2.flip(process_image_for_unet, 0)
    pred_cobb, pred_num_vertebrae = calculate_cobb_angle(pred_mask, label_offset=0)
    print(f"[process_image] Cobb angle calculated: {pred_cobb}, vertebrae count: {pred_num_vertebrae}")
    inference_time = time.time() - start_time
    gt_mask = None
    gt_cobb = None
    gt_num_vertebrae = 0
    if source_type == "xray":
        gt_cobb = gt_angle_from_filename
        print(f"[process_image] GT angle from filename: {gt_cobb}")
    elif gt_mask_path and os.path.exists(gt_mask_path):
        print(f"[process_image] Reading GT mask: {gt_mask_path}")
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"[process_image] WARNING: GT mask could not be read! {gt_mask_path}")
        else:
            gt_mask = cv2.resize(
                gt_mask,
                (pred_mask.shape[1], pred_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            if source_type == "hospital":
                gt_mask = cv2.flip(gt_mask, 0)
                label_offset = 0
            elif source_type == "totalsegmentator":
                label_offset = 7
            else:
                label_offset = 0
            gt_cobb, gt_num_vertebrae = calculate_cobb_angle(gt_mask, label_offset=label_offset)
            print(f"[process_image] GT Cobb angle: {gt_cobb}, vertebrae count: {gt_num_vertebrae}")
    
    gallery_images = []
    
    if source_type in ["xray", "unknown"]:
        gray_display = cv2.cvtColor(original_input, cv2.COLOR_RGB2GRAY)
        title = "Original X-Ray" if source_type == "xray" else "Input Image"
        gallery_images.append((gray_display, title))
        gallery_images.append((fake_drr_np, "CycleGAN DRR"))
        
        seg_colored = plt.cm.tab20(pred_mask / 17.0)[:, :, :3]
        seg_colored = (seg_colored * 255).astype(np.uint8)
        seg_title = "Segmentation"
        gallery_images.append((seg_colored, seg_title))
        
        if pred_cobb is not None:
            overlay = cv2.addWeighted(fake_drr_np, 0.5, seg_colored, 0.5, 0)
            if gt_cobb is not None:
                error = abs(pred_cobb - gt_cobb)
                title = f"Pred: {pred_cobb:.1f} | GT: {gt_cobb:.1f} | Err: {error:.1f}"
            else:
                title = f"Prediction: {pred_cobb:.1f}"
            gallery_images.append((overlay, title))
        else:
            gallery_images.append((fake_drr_np, "Insufficient vertebrae"))
    else:
        source_name = "Hospital DRR" if source_type == "hospital" else "TotalSegmentator DRR"
        gallery_images.append((process_image_for_unet, source_name))
        
        if gt_mask is not None:
            vmax = 24 if source_type == "totalsegmentator" else 17
            # Use GT mask for visualization (it was flipped, flip back)
            gt_mask_display = cv2.flip(gt_mask, 0) if source_type == "hospital" else gt_mask
            gt_colored = plt.cm.tab20(gt_mask_display / vmax)[:, :, :3]
            gt_colored = (gt_colored * 255).astype(np.uint8)
            gt_title = f"GT Cobb: {gt_cobb:.1f}" if gt_cobb else "Ground Truth"
            gallery_images.append((gt_colored, gt_title))
        
        seg_colored = plt.cm.tab20(pred_mask / 17.0)[:, :, :3]
        seg_colored = (seg_colored * 255).astype(np.uint8)
        pred_title = f"Prediction: {pred_cobb:.1f}" if pred_cobb else "Model Prediction"
        gallery_images.append((seg_colored, pred_title))
        
        if pred_cobb is not None:
            overlay = cv2.addWeighted(process_image_for_unet, 0.5, seg_colored, 0.5, 0)
            if gt_cobb is not None:
                error = abs(pred_cobb - gt_cobb)
                title = f"Error: {error:.1f}"
            else:
                title = "Overlay"
            gallery_images.append((overlay, title))
    
    source_names = {
        'xray': 'X-Ray (via CycleGAN)',
        'hospital': 'Hospital DRR',
        'totalsegmentator': 'TotalSegmentator DRR',
        'unknown': 'Unknown (via CycleGAN)'
    }
    
    results = "## Analysis Results\n\n"
    results += "| Property | Value |\n|---|---|\n"
    results += f"| **Source Type** | {source_names.get(source_type, source_type)} |\n"
    
    if pred_cobb is not None:
        results += f"| **Predicted Cobb Angle** | **{pred_cobb:.2f}** |\n"
        
        if gt_cobb is not None:
            error = abs(pred_cobb - gt_cobb)
            results += f"| **Ground Truth Angle** | {gt_cobb:.2f} |\n"
            results += f"| **Absolute Error** | {error:.2f} |\n"
            
            if gt_mask is not None:
                # Multi-class Dice: calculate for each vertebra individually
                dice_scores = []
                unique_labels = np.unique(gt_mask)
                unique_labels = unique_labels[unique_labels > 0]  # Exclude background
                
                for label in unique_labels:
                    gt_binary = (gt_mask == label).astype(np.uint8)
                    pred_binary = (pred_mask == label).astype(np.uint8)
                    intersection = np.sum(gt_binary & pred_binary)
                    union = np.sum(gt_binary) + np.sum(pred_binary)
                    if union > 0:
                        dice_scores.append(2 * intersection / union)
                
                dice = np.mean(dice_scores) if dice_scores else 0.0
                results += f"| **Dice Score (Multi-class)** | {dice:.4f} |\n"
    else:
        results += "| **Status** | Insufficient vertebrae detected |\n"
    
    results += f"| **Inference Time** | {inference_time:.2f}s |\n"
    
    if filepath:
        results += f"\n**File:** `{os.path.basename(filepath)}`\n"
    
    return gallery_images, results


def load_image_from_path(filepath):
    """Load image from file path."""
    if filepath is None or filepath.strip() == "":
        return None, filepath
    
    filepath = filepath.strip().strip('"').strip("'")
    
    if os.path.exists(filepath):
        img = cv2.imread(filepath)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"[load_image_from_path] File loaded: {filepath}")
            return img, filepath
    
    return None, filepath


def clear_results():
    """Clear filepath and results when image is changed/removed."""
    return "", [], ""


def handle_image_upload(image):
    """Handle direct image upload (drag/drop or file select).
    
    Since Gradio doesn't provide full filepath for security reasons,
    the image will be processed as 'unknown' (using CycleGAN).
    """
    if image is None:
        print("[handle_image_upload] Image removed.")
        return "", [], "Image removed. Upload again or enter file path."
    
    print("[handle_image_upload] Image uploaded directly. Processing as 'Unknown'.")
    info_msg = """
    **Image Uploaded!**
    
    Processing as **'Unknown'** (using CycleGAN).
    
    **For GT mask comparison:** Paste full file path in **File Path** box and click **Load**.
    """
    return "", [], info_msg


def build_interface():
    """Build Gradio web interface."""
    with gr.Blocks(title="Cobb Angle Measurement") as demo:
        gr.Markdown("""
        # Cobb Angle Measurement System
        
        Upload an image or enter a file path to measure spine Cobb angle.
        
        **Automatic Detection:**
        - `hospital_drr/` - Hospital DRR (uses ground truth mask)
        - `totalsegmentator_drr/` - TotalSegmentator DRR (uses ground truth mask with label offset)
        - `labeled_xray/` or files starting with `sunhl` - X-Ray (CycleGAN applied)
        - Other - Processed via CycleGAN
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                filepath_input = gr.Textbox(
                    label="File Path",
                    placeholder="e.g., data/hospital_drr/trainCT/AA.png",
                    lines=2
                )
                image_input = gr.Image(type="numpy", label="Image", height=350)
                
                with gr.Row():
                    load_btn = gr.Button("Load", variant="secondary")
                    analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                
            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Results (click to zoom)",
                    columns=4,
                    height=450,
                    object_fit="contain"
                )
                output_text = gr.Markdown()
        
        gr.Markdown("### Example Paths")
        with gr.Row():
            gr.Textbox(
                value="data/hospital_drr/trainCT/AA.png",
                label="Hospital DRR",
                interactive=False
            )
            gr.Textbox(
                value="data/totalsegmentator_drr/trainCT/case_0000.png",
                label="TotalSegmentator DRR",
                interactive=False
            )
            gr.Textbox(
                value="data/labeled_xray/sunhl-1th-06-Jan-2017-187 A AP_gt11.5.jpg",
                label="X-Ray",
                interactive=False
            )
        
        load_btn.click(
            fn=load_image_from_path,
            inputs=[filepath_input],
            outputs=[image_input, filepath_input]
        )
        filepath_input.submit(
            fn=load_image_from_path,
            inputs=[filepath_input],
            outputs=[image_input, filepath_input]
        )
        analyze_btn.click(
            fn=process_image,
            inputs=[image_input, filepath_input],
            outputs=[output_gallery, output_text]
        )
        
        # Show info message when image is uploaded directly
        image_input.upload(
            fn=handle_image_upload,
            inputs=[image_input],
            outputs=[filepath_input, output_gallery, output_text]
        )
        
        # Clear results when image changes or is cleared
        image_input.clear(
            fn=clear_results,
            inputs=[],
            outputs=[filepath_input, output_gallery, output_text]
        )
    
    return demo


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    
    if load_models():
        print("Models loaded successfully")
        app = build_interface()
        app.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            inbrowser=True
        )
    else:
        print("Failed to load models")
