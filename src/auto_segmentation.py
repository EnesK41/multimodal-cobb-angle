# Import necessary libraries
import subprocess
import os
import sys
import glob  # To list files in a directory

def run_segmentation(input_file, output_file, task="vertebrae", fast_mode=True):
    """
    Runs TotalSegmentator automatically from the command line for a single file.
    """
    print(f"\n--> Processing: {os.path.basename(input_file)}")

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return False

    command = [
        "TotalSegmentator", "-i", input_file, "-o", output_file, "--task", task
    ]
    if fast_mode:
        command.append("--fast")

    try:
        subprocess.run(command, check=True, encoding='utf-8')
        print(f"--> SUCCESS: {os.path.basename(output_file)} created.")
        return True

    except subprocess.CalledProcessError:
        print(f"--> ERROR during {os.path.basename(input_file)}!")
        return False
    except Exception as e:
        print(f"--> UNEXPECTED ERROR: {e}")
        return False

if __name__ == "__main__":
    
    input_folder = "../tobesegmented"
    output_folder = "../segmented_masks"
    task_name = "vertebrae_mr"
    use_fast_mode = False

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all .nii.gz files in the input folder
    nifti_files = glob.glob(os.path.join(input_folder, "*.nii.gz"))
    
    print(f"=== TOTAL AUTOMATION STARTED ===")
    print(f"Found files: {len(nifti_files)}")
    print(f"Task: {task_name}, Fast Mode: {use_fast_mode}\n")

    success_count = 0
    error_count = 0

    # Loop through each file
    for input_path in nifti_files:
        file_name = os.path.basename(input_path)
        output_name = file_name.replace(".nii.gz", "_mask.nii.gz")
        output_path = os.path.join(output_folder, output_name)

        # Skip if mask already exists
        if os.path.exists(output_path):
            print(f"SKIPPING: {output_name} already exists.")
            success_count += 1
            continue

        success = run_segmentation(input_path, output_path, task=task_name, fast_mode=use_fast_mode)
        
        if success:
            success_count += 1
        else:
            error_count += 1

    print(f"\n=== AUTOMATION COMPLETED ===")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
