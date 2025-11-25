import torch
import torch.nn as nn
import torchvision.models as models

class MedicalEncoder(nn.Module):
    """
    Feature Extractor model based on ResNet-18.
    It adapts the standard ResNet to accept grayscale medical images (1 channel)
    instead of RGB images (3 channels).
    """
    def __init__(self, in_channels=1, pretrained=True):
        super(MedicalEncoder, self).__init__()
        
        # 1. Load the pre-trained ResNet18 model
        # pretrained=True means we download weights trained on ImageNet.
        # This gives the model a "head start" in understanding visual features.
        self.model = models.resnet18(pretrained=pretrained)
        
        # 2. Modify the first layer (Input Layer)
        # Standard ResNet expects 3 channels (RGB). We change it to 'in_channels' (usually 1 for X-ray/CT).
        # We keep the other parameters (weights/bias) similar to preserve learned features.
        original_first_layer = self.model.conv1
        
        self.model.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=original_first_layer.out_channels, 
            kernel_size=original_first_layer.kernel_size, 
            stride=original_first_layer.stride, 
            padding=original_first_layer.padding, 
            bias=False
        )
        
        # 3. Remove the Classification Head
        # We need feature maps, not a classification (cat/dog).
        # So we remove the last two layers: Average Pooling (avgpool) and Fully Connected (fc).
        # The output will be a spatial feature map (e.g., 512 channels x 16 height x 16 width).
        self.backbone = nn.Sequential(*list(self.model.children())[:-2]) 

    def forward(self, x):
        # Pass the input 'x' through the modified backbone
        features = self.backbone(x)
        return features

class LandmarkHead(nn.Module):
    """
    Predictor Head designed to take feature maps from the Encoder
    and predict heatmap locations for landmarks.
    """
    def __init__(self, input_channels=512, num_landmarks=68):
        super(LandmarkHead, self).__init__()
        
        # A simple convolutional block to process the features
        # It reduces channels and prepares the output map.
        self.head = nn.Sequential(
            # Conv Layer 1: Refine features
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # Normalize for stability
            nn.ReLU(inplace=True), # Activation function
            
            # Conv Layer 2: Output layer
            # Produces 'num_landmarks' channels (one heatmap per landmark point)
            # Kernel size 1x1 acts as a per-pixel fully connected layer.
            nn.Conv2d(256, num_landmarks, kernel_size=1) 
        )

    def forward(self, x):
        return self.head(x)

class EndToEndModel(nn.Module):
    """
    The full model combining the Encoder and the Landmark Head.
    """
    def __init__(self):
        super(EndToEndModel, self).__init__()
        self.encoder = MedicalEncoder()
        self.head = LandmarkHead()
    
    def forward(self, x):
        features = self.encoder(x)
        output = self.head(features)
        return output

# --- DUMMY DATA TEST BLOCK (Run this file to test without real data) ---
if __name__ == "__main__":
    print("--- Testing Model Architecture ---")
    
    # 1. Initialize the full model
    model = EndToEndModel()
    print("Model initialized successfully.")

    # 2. Create 'Dummy Data' (Fake X-ray)
    # Format: (Batch Size, Channels, Height, Width)
    # Batch=2, Channel=1 (Grayscale), 512x512 resolution
    dummy_input = torch.randn(2, 1, 512, 512)
    print(f"Input shape: {dummy_input.shape}")

    # 3. Feed the dummy data to the model (Forward Pass)
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        
        # 4. Verify the output dimensions
        # Expected: (2, 68, 16, 16) -> 2 images, 68 landmarks, 16x16 heatmap size
        if output.shape == (2, 68, 16, 16):
            print("✅ SUCCESS: Dimensions match expected output.")
        else:
            print("⚠️ WARNING: Dimensions are different than expected.")
            
    except Exception as e:
        print(f"❌ ERROR: Model failed during forward pass.\n{e}")