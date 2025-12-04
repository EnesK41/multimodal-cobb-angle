import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

# --- AYARLAR ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_DIR = os.path.join(BASE_DIR, "data", "test_real_xray")

def prepare_data(source_dir, images_dir=None):
    """
    AASCE veri setini işleyip etiketli test verisi oluşturur.
    
    Args:
        source_dir: angles.csv ve filenames.csv'nin bulunduğu ana klasör
        images_dir: Resimlerin bulunduğu klasör (belirtilmezse source_dir/train kullanılır)
    """
    print("🚀 Test Verisi Hazırlanıyor...")
    
    if images_dir is None:
        images_dir = os.path.join(source_dir, "train")
    
    csv_path = os.path.join(source_dir, "train_txt", "angles.csv")
    
    if os.path.exists(DEST_DIR): shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR, exist_ok=True)

    try:
        names_path = os.path.join(source_dir, "train_txt", "filenames.csv")
        
        if os.path.exists(names_path):
            df_names = pd.read_csv(names_path, header=None)
            df_angles = pd.read_csv(csv_path, header=None)
            
            if len(df_names) != len(df_angles):
                print("⚠️ Uyarı: İsim ve Açı dosyası satır sayısı uyuşmuyor!")
            
            print(f"📄 {len(df_names)} dosya işleniyor...")

            count = 0
            for i in range(len(df_names)):
                filename = df_names.iloc[i, 0]
                angles = df_angles.iloc[i, :].values.astype(float)
                cobb_angle = max(angles)
                
                src_path = os.path.join(images_dir, filename)
                if not src_path.endswith(".jpg"): src_path += ".jpg"
                
                if not os.path.exists(src_path): continue

                new_name = f"{os.path.splitext(filename)[0]}_gt{cobb_angle:.1f}.jpg"
                shutil.copy(src_path, os.path.join(DEST_DIR, new_name))
                count += 1
                
            print(f"✅ {count} adet etiketli test verisi hazır!")
            
        else:
            print("❌ filenames.csv bulunamadı!")

    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AASCE veri setini etiketli test verisine dönüştürür")
    parser.add_argument("--source", "-s", required=True, 
                        help="AASCE veri setinin ana klasörü (angles.csv'nin bulunduğu yer)")
    parser.add_argument("--images", "-i", default=None,
                        help="Resimlerin bulunduğu klasör (varsayılan: source/train)")
    
    args = parser.parse_args()
    prepare_data(args.source, args.images)