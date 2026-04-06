import os
import cv2
import json
import numpy as np
import random
from augraphy import (
    AugraphyPipeline,
    NoiseTexturize,
    Brightness,
    LightingGradient,
    Gamma,
)

# ── Augraphy pipeline ──────────────────────────────────────────────────────────
ancient_pipeline = [
    NoiseTexturize(sigma_range=(3, 15), turbulence_range=(2, 5), p=1.0),
    Brightness(brightness_range=(0.5, 1.5), p=1.0),
    LightingGradient(max_brightness=192, p=0.7),
    Gamma(gamma_range=(0.5, 2.0), p=1.0),
]

pipeline = AugraphyPipeline(ink_phase=[], paper_phase=[], post_phase=ancient_pipeline)

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_eroded_mask(gray_text: np.ndarray, erosion_strength: float = 0) -> np.ndarray:
    """
    Create a soft, slightly-eroded mask that mimics weathered carved edges.
    Returns a float32 mask in [0, 1].
    """
    # Adjust this 120 threshold if you want globally thinner/thicker text before erosion
    _, hard_mask = cv2.threshold(gray_text, 120, 255, cv2.THRESH_BINARY_INV)

    if erosion_strength > 0:
        # Convert float to nearest int so OpenCV doesn't crash!
        e_str = int(round(erosion_strength))
        if e_str > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (e_str, e_str)
            )
            hard_mask = cv2.erode(hard_mask, kernel, iterations=1)

    soft_mask = cv2.GaussianBlur(hard_mask, (5, 5), 0).astype(np.float32) / 255.0
    return soft_mask

def add_chisel_shadow(
    base: np.ndarray, mask: np.ndarray, shadow_offset: tuple = (2, 2),
    shadow_strength: float = 0.5, highlight_strength: float = 0.3,
) -> np.ndarray:
    result = base.astype(np.float32)
    h, w = mask.shape
    sx, sy = shadow_offset

    shadow_mask = np.zeros_like(mask)
    shadow_mask[max(sy, 0) : h, max(sx, 0) : w] = mask[: h - max(sy, 0), : w - max(sx, 0)]
    result -= shadow_mask[:, :, np.newaxis] * shadow_strength * 255

    hi_mask = np.zeros_like(mask)
    hi_mask[: h - max(sy, 0), : w - max(sx, 0)] = mask[max(sy, 0) : h, max(sx, 0) : w]
    result += hi_mask[:, :, np.newaxis] * highlight_strength * 255

    return np.clip(result, 0, 255).astype(np.uint8)

def blend_text_onto_stone(
    stone: np.ndarray, soft_mask: np.ndarray, dark_factor: float, mean_color: tuple,
) -> np.ndarray:
    tint = np.array(mean_color, dtype=np.float32)
    etched = cv2.convertScaleAbs(stone, alpha=dark_factor, beta=0).astype(np.float32)
    etched = etched * 0.8 + tint[np.newaxis, np.newaxis, :] * 0.2
    etched = np.clip(etched, 0, 255).astype(np.uint8)

    alpha = soft_mask[:, :, np.newaxis]
    combined = (stone.astype(np.float32) * (1 - alpha) + etched.astype(np.float32) * alpha)
    return np.clip(combined, 0, 255).astype(np.uint8)

def create_ancient_variation(
    text_img_path: str, stone_img_path: str, output_path: str,
    pipeline_obj: AugraphyPipeline, dark_factor: float,
    erosion_strength: float, shadow_offset: tuple
) -> bool:
    text_img  = cv2.imread(text_img_path)
    stone_img = cv2.imread(stone_img_path)

    if text_img is None or stone_img is None:
        print(f"Warning: Could not load '{text_img_path}' or '{stone_img_path}'. Skipping.")
        return False

    stone_img = cv2.resize(stone_img, (text_img.shape[1], text_img.shape[0]))
    gray_text  = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    soft_mask  = make_eroded_mask(gray_text, erosion_strength)
    mean_color = cv2.mean(stone_img)[:3]

    combined = blend_text_onto_stone(stone_img, soft_mask, dark_factor, mean_color)
    combined = add_chisel_shadow(combined, soft_mask, shadow_offset)
    combined = cv2.GaussianBlur(combined, (3, 3), 0)

    augmented = pipeline_obj.augment(combined)
    if isinstance(augmented, dict):
        augmented = augmented.get("output", combined)

    cv2.imwrite(output_path, augmented)
    return True

# ── JSONL Processing Pipeline ──────────────────────────────────────────────────

def process_dataset_from_jsonl(
    input_jsonl: str, output_jsonl: str, output_img_dir: str, stone_files: list,
    dark_factor: float, erosion_strength: float, shadow_offset: tuple
):
    os.makedirs(output_img_dir, exist_ok=True)
    
    print(f"Reading dataset from: {input_jsonl}")
    
    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:
         
        for line_idx, line in enumerate(infile):
            record = json.loads(line.strip())
            
            # 1. Find the image path in the JSON record
            original_img_path = None
            for content_item in record["messages"][0]["content"]:
                if content_item["type"] == "image":
                    original_img_path = content_item["image"]
                    break
            
            if not original_img_path:
                print(f"Skipping line {line_idx}: No image found in JSON.")
                continue
                
            # 2. Set up the new file path for the stone image
            base_name = os.path.basename(original_img_path)
            new_img_path = os.path.join(output_img_dir, f"stone_{base_name}")
            
            # 3. Choose a random stone background
            stone_bg = random.choice(stone_files)
            
            # 4. Process the image
            success = create_ancient_variation(
                original_img_path, stone_bg, new_img_path, pipeline,
                dark_factor=dark_factor, 
                erosion_strength=erosion_strength, 
                shadow_offset=shadow_offset
            )
            
            # 5. Update the JSON record with the new path and save it
            if success:
                for content_item in record["messages"][0]["content"]:
                    if content_item["type"] == "image":
                        # Update the path to point to the new stone image
                        content_item["image"] = new_img_path
                        break
                
                # Write the updated line to the new JSONL file
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f"Processed: {new_img_path}")

    print(f"\nFinished! Updated dataset saved to: {output_jsonl}")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # --- PATHS ---
    # The JSONL generated by your previous text-to-image script
    INPUT_JSONL = r"C:\Users\vigne\OneDrive\Documents\LIU assignments\NLP ass\Data\Images\chunked_images\vlm_dataset.jsonl"
    
    # Where the final JSONL and stone images should be saved
    OUTPUT_JSONL = r"C:\Users\vigne\OneDrive\Documents\LIU assignments\NLP ass\Data\Images\chunked_images\final_vlm_dataset.jsonl"
    OUTPUT_IMG_DIR = r"C:\Users\vigne\OneDrive\Documents\LIU assignments\NLP ass\Data\Images\chunked_images\stone_images"
    
    # List of all available background stone images
    STONE_FILES = [
        r"Code\bgs\image.png",
        r"Code\bgs\Gemini_Generated_Image_3j8n2p3j8n2p3j8n.png",
        r"Code\bgs\Gemini_Generated_Image_51lysy51lysy51ly.png",
        r"Code\bgs\Gemini_Generated_Image_cf26iscf26iscf26.png",
        r"Code\bgs\istockphoto-936307734-612x612.jpg",
        r"Code\bgs\pexels-sora-noao-265549236-12998761.jpg",
        r"Code\bgs\photo-wall-texture-pattern.jpg",
        r"Code\bgs\rock-wall-bearing-numerous-carved-inscriptions-and-symbols-credit-ministry-of-tourism-and-antiquities.jpg"
        # Add more paths here if you have multiple stone backgrounds:
        # r"Code\bgs\stone_2.png",
    ]

    # --- HYPERPARAMETERS ---
    MANUAL_DARK_FACTOR = 0.50 
    MANUAL_EROSION_STRENGTH = 1 # You can now safely use floats!
    MANUAL_SHADOW_OFFSET = (2, 2) 

    # --- RUN PIPELINE ---
    process_dataset_from_jsonl(
        input_jsonl=INPUT_JSONL,
        output_jsonl=OUTPUT_JSONL,
        output_img_dir=OUTPUT_IMG_DIR,
        stone_files=STONE_FILES,
        dark_factor=MANUAL_DARK_FACTOR,
        erosion_strength=MANUAL_EROSION_STRENGTH,
        shadow_offset=MANUAL_SHADOW_OFFSET
    )