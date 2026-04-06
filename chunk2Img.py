import os
import glob
import cv2
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def make_text_rough_and_wobbly(pil_image, wobble_amount=3.0, roughness=60, edge_blur=5, solidify_kernel=3):
    """Applies elastic distortion, jagged edges, and solidifies porous text."""
    img = np.array(pil_image.convert('L'))
    h, w = img.shape

    # 1. THE WOBBLE
    noise_x = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (15, 15), 5) * wobble_amount
    noise_y = cv2.GaussianBlur(np.random.randn(h, w).astype(np.float32), (15, 15), 5) * wobble_amount
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.float32(x + noise_x)
    map_y = np.float32(y + noise_y)
    
    img_wobbled = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    # 2. THE JAGGED EDGE
    img_blur = cv2.GaussianBlur(img_wobbled, (edge_blur, edge_blur), 0)
    static_noise = np.random.randint(-roughness, roughness, (h, w)).astype(np.int16)
    noisy_blur = np.clip(img_blur.astype(np.int16) + static_noise, 0, 255).astype(np.uint8)
    _, final_img = cv2.threshold(noisy_blur, 128, 255, cv2.THRESH_BINARY)
    
    # --- 3. THE SOLIDIFIER ---
    binary_img_inv = cv2.bitwise_not(final_img)
    kernel = np.ones((solidify_kernel, solidify_kernel), np.uint8)
    solidified_inv = cv2.morphologyEx(binary_img_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    final_solid_img = cv2.bitwise_not(solidified_inv)
    
    return Image.fromarray(final_solid_img)

def extract_chunks_from_file(filepath):
    """Parses a text file and extracts chunks, preserving their original line breaks."""
    chunks = []
    current_chunk_lines = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("--- Chunk") and line.endswith("---"):
                if current_chunk_lines:
                    chunks.append("\n".join(current_chunk_lines))
                    current_chunk_lines = []
            elif line: 
                current_chunk_lines.append(line)
                
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))
        
    return chunks

def generate_inscription_images(chunks, font_path, output_dir, base_filename, jsonl_filepath, font_size=60, wobble_amount=4.0, roughness=70, edge_blur=5, solidify_kernel=3):
    """Renders text chunks to images exactly as formatted, and logs to JSONL."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Could not load font at {font_path}. Check the path.")
        return

    # Open the JSONL file in append mode
    with open(jsonl_filepath, 'a', encoding='utf-8') as jsonl_file:
        for i, chunk_text in enumerate(chunks):
            joined_text = chunk_text

            # Dynamic size calculation
            dummy_img = Image.new('RGB', (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)
            bbox = dummy_draw.multiline_textbbox((0, 0), joined_text, font=font, spacing=15)
            
            padding = 60
            img_w = bbox[2] - bbox[0] + padding * 2
            img_h = bbox[3] - bbox[1] + padding * 2

            # Draw pristine text respecting original newlines
            img = Image.new('RGB', (img_w, img_h), color='white')
            draw = ImageDraw.Draw(img)
            draw.multiline_text((padding, padding), joined_text, fill="black", font=font, spacing=15)

            # Apply rough filters
            final_image = make_text_rough_and_wobbly(
                img, 
                wobble_amount=wobble_amount, 
                roughness=roughness, 
                edge_blur=edge_blur, 
                solidify_kernel=solidify_kernel
            )

            # Save the image
            image_filename = f"{base_filename}_chunk_{i+1}.png"
            output_filepath = os.path.join(output_dir, image_filename)
            final_image.save(output_filepath)
            
            # --- CREATE JSONL ENTRY FOR VLM TRAINING ---
            vlm_record = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": output_filepath},
                            {"type": "text", "text": "Transcribe the ancient text in this image."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": chunk_text}
                        ]
                    }
                ]
            }
            jsonl_file.write(json.dumps(vlm_record, ensure_ascii=False) + '\n')

# --- Run the pipeline ---
if __name__ == "__main__":
    
    # --- UPDATE THIS PATH TO YOUR TXT FOLDER ---
    input_folder = r"C:\Users\vigne\OneDrive\Documents\LIU assignments\NLP ass\Data\Raw txt\Modified\whole_chunked"  
    
    font_file_tamili = r"Code\e-Brahmi-T.ttf"
    font_file_vatteluttu = r"Code\e-VatteluttuOT.ttf"
    
    # Set up master output directory and subdirectories
    output_folder_root = r"C:\Users\vigne\OneDrive\Documents\LIU assignments\NLP ass\Data\Images\chunked_images"
    output_folder_tamili = os.path.join(output_folder_root, "tamili")
    output_folder_vatteluttu = os.path.join(output_folder_root, "vatteluttu")
    
    # The single master JSONL file sitting in the root output folder
    jsonl_path = os.path.join(output_folder_root, "vlm_dataset.jsonl") 
    
    # Clear old JSONL file to prevent duplicate entries
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)
        
    # Ensure the root folder exists before making the json file
    if not os.path.exists(output_folder_root):
        os.makedirs(output_folder_root)
    
    text_files = glob.glob(os.path.join(input_folder, "*.txt"))
    
    if not text_files:
        print(f"No .txt files found in {input_folder}. Please check the path.")
    else:
        print(f"Found {len(text_files)} text files. Generating images and building master JSONL dataset...")
        
        for filepath in text_files:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            print(f"Processing: {base_name}.txt")
            
            data_chunks = extract_chunks_from_file(filepath)
            
            if not data_chunks:
                print(f" -> No chunks found in {base_name}. Skipping.")
                continue
            
            print(f" -> Generating {len(data_chunks)} Tamili and {len(data_chunks)} Vatteluttu images...")
            
            # --- GENERATE TAMILI IMAGES ---
            generate_inscription_images(
                chunks=data_chunks, 
                font_path=font_file_tamili, 
                output_dir=output_folder_tamili, 
                base_filename=f"{base_name}_tamili", # Differentiating the filenames
                jsonl_filepath=jsonl_path, 
                font_size=40, 
                wobble_amount=2.0, 
                roughness=50,      
                edge_blur=1,       
                solidify_kernel=2 
            )
            
            # --- GENERATE VATTELUTTU IMAGES ---
            generate_inscription_images(
                chunks=data_chunks, 
                font_path=font_file_vatteluttu, 
                output_dir=output_folder_vatteluttu, 
                base_filename=f"{base_name}_vatteluttu", # Differentiating the filenames
                jsonl_filepath=jsonl_path, 
                font_size=40, 
                wobble_amount=2.0, 
                roughness=50,      
                edge_blur=1,       
                solidify_kernel=1 
            )
        
        print(f"\nSuccess! Images saved to '{output_folder_tamili}' and '{output_folder_vatteluttu}'.")
        print(f"Master dataset mapped safely to: {jsonl_path}")