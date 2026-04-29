import os
import glob
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

def generate_masks(input_dir, output_human_dir, output_bg_dir):
    """
    Processes a directory of images to generate human and background segmentation masks.
    """
    os.makedirs(output_human_dir, exist_ok=True)
    os.makedirs(output_bg_dir, exist_ok=True)

    print("Loading YOLOv8 segmentation model...")
    model = YOLO('yolov8n-seg.pt')

    image_paths = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images to process in {input_dir}.")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        h, w = img.shape[:2]

        # retina_masks=True ensures masks are output at the original image resolution
        results = model(img, verbose=False, retina_masks=True)
        
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for result in results:
            if result.masks is not None:
                masks_data = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for i, cls in enumerate(classes):
                    if int(cls) == 0:  # Class 0 is 'person'
                        mask = masks_data[i]
                        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)

        human_mask = combined_mask * 255
        bg_mask = 255 - human_mask

        cv2.imwrite(os.path.join(output_human_dir, filename), human_mask)
        cv2.imwrite(os.path.join(output_bg_dir, filename), bg_mask)

    print("Mask generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLOv8 segmentation masks.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input frames.")
    parser.add_argument("--output_human_dir", type=str, required=True, help="Directory to save human masks.")
    parser.add_argument("--output_bg_dir", type=str, required=True, help="Directory to save background masks.")
    
    args = parser.parse_args()
    generate_masks(args.input_dir, args.output_human_dir, args.output_bg_dir)