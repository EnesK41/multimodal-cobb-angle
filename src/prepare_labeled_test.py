import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIR = os.path.join(BASE_DIR, "data", "test_real_xray")

def prepare_data(source_dir, images_dir=None):
    """
    Process AASCE dataset and create labeled test data.
    
    Args:
        source_dir: Root folder containing angles.csv and filenames.csv
        images_dir: Folder containing images (defaults to source_dir/train)
    """
    print("🚀 Preparing test data...")
    
    if images_dir is None:
        images_dir = os.path.join(source_dir, "train")
    
    csv_path = os.path.join(source_dir, "train_txt", "angles.csv")
    
    if os.path.exists(DEST_DIR):
        shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR, exist_ok=True)

    try:
        names_path = os.path.join(source_dir, "train_txt", "filenames.csv")
        
        if os.path.exists(names_path):
            df_names = pd.read_csv(names_path, header=None)
            df_angles = pd.read_csv(csv_path, header=None)
            
            if len(df_names) != len(df_angles):
                print("⚠️ Warning: Filename and angle file row counts don't match!")
            
            print(f"📄 Processing {len(df_names)} files...")

            count = 0
            for i in range(len(df_names)):
                filename = df_names.iloc[i, 0]
                angles = df_angles.iloc[i, :].values.astype(float)
                cobb_angle = max(angles)
                
                src_path = os.path.join(images_dir, filename)
                if not src_path.endswith(".jpg"):
                    src_path += ".jpg"
                
                if not os.path.exists(src_path):
                    continue

                new_name = f"{os.path.splitext(filename)[0]}_gt{cobb_angle:.1f}.jpg"
                shutil.copy(src_path, os.path.join(DEST_DIR, new_name))
                count += 1
                
            print(f"✅ {count} labeled test images ready!")
            
        else:
            print("❌ filenames.csv not found!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert AASCE dataset to labeled test data")
    parser.add_argument("--source", "-s", required=True, 
                        help="AASCE dataset root folder (containing angles.csv)")
    parser.add_argument("--images", "-i", default=None,
                        help="Images folder (default: source/train)")
    
    args = parser.parse_args()
    prepare_data(args.source, args.images)