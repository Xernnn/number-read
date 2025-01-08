import cv2
import numpy as np
import os

def extract_digits(input_folder, output_folder):
    """Extract individual digits from images and save them"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if not filename.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Get the true label from filename
        true_chars = []
        for char in filename.split('.')[0]:
            if char.isdigit() or char in [':', '.', ',', '-']:  # Include special characters
                true_chars.append(char)
        
        if not true_chars:
            continue
            
        print(f"Processing {filename}...")
        
        # Read and preprocess image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {filename}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Debug images
        debug_original = img.copy()
        debug_binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Filter and sort components
        valid_components = []
        min_height = height // 4  # Keep this for sensitivity
        expected_width = width // len(true_chars)
        
        # First pass: collect all potential components
        potential_components = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            aspect_ratio = h/w if w > 0 else 0
            
            # Calculate component properties
            roi = binary[y:y+h, x:x+w]
            avg_intensity = np.mean(roi)
            fill_ratio = area / (w * h)
            
            # Even more relaxed criteria for digits
            is_digit = (
                h >= min_height * 0.4 and         
                0.05 * width/len(true_chars) <= w <= 2.5 * width/len(true_chars) and  
                0.15 <= aspect_ratio <= 4.0 and   
                (fill_ratio > 0.08 or             
                 (area > 30 and aspect_ratio > 1.5)) and  
                avg_intensity > 30                
            )
            
            # Special cases for specific digits
            is_special = (
                h >= min_height * 0.4 and
                w <= expected_width * 0.8 and    
                area > 30 and                     
                (aspect_ratio >= 0.8 or          
                 fill_ratio > 0.15)              
            )
            
            # Dash detection criteria
            is_dash = (
                h < min_height * 0.5 and         # Must be shorter than digits
                w > expected_width * 0.2 and     # Must be reasonably wide
                aspect_ratio < 0.5 and           # Must be wider than tall
                fill_ratio > 0.3 and            # Must be solid
                avg_intensity > 30 and
                y < height * 0.6                # Usually in upper half
            )
            
            is_digit = is_digit or is_special
            
            if is_digit:
                potential_components.append((x, y, w, h, area, 'digit'))
            elif is_dash:
                potential_components.append((x, y, w, h, area, 'dash'))
        
        # Sort by x-coordinate
        potential_components.sort(key=lambda x: x[0])
        
        # Second pass: merge components that might be split digits (like 4)
        i = 0
        while i < len(potential_components):
            current = potential_components[i]
            
            # If this isn't the last component
            if i < len(potential_components) - 1:
                next_comp = potential_components[i + 1]
                
                # Calculate gaps and overlaps
                gap = next_comp[0] - (current[0] + current[2])
                vertical_overlap = (min(current[1] + current[3], next_comp[1] + next_comp[3]) - 
                                 max(current[1], next_comp[1]))
                
                # Calculate widths relative to expected
                current_width_ratio = current[2] / expected_width
                next_width_ratio = next_comp[2] / expected_width
                combined_width = next_comp[0] + next_comp[2] - current[0]
                combined_width_ratio = combined_width / expected_width
                
                # Merge only if:
                # 1. Components are very close
                # 2. Combined width isn't too large
                # 3. Has vertical overlap
                # 4. Individual components are thin
                should_merge = (
                    gap < expected_width * 0.2 and           # Close enough
                    combined_width_ratio < 1.5 and           # Not too wide when combined
                    vertical_overlap > min(current[3], next_comp[3]) * 0.3 and  # Significant vertical overlap
                    (current_width_ratio < 0.5 or next_width_ratio < 0.5)  # At least one is thin
                )
                
                if should_merge:
                    # Merge components
                    x = current[0]
                    y = min(current[1], next_comp[1])
                    w = next_comp[0] + next_comp[2] - current[0]
                    h = max(current[1] + current[3], next_comp[1] + next_comp[3]) - y
                    area = current[4] + next_comp[4]
                    valid_components.append((x, y, w, h, 'digit'))
                    i += 2
                    continue
            
            valid_components.append(current[:4] + ('digit',))
            i += 1
        
        # Draw debug visualization
        for x, y, w, h, comp_type in valid_components:
            if comp_type == 'digit':
                color = (0, 255, 0)    # Green for digits
            elif comp_type == 'dash':
                color = (0, 0, 255)    # Red for dash
            else:
                color = (255, 0, 0)    # Blue for others
                
            cv2.rectangle(debug_original, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(debug_binary, (x, y), (x+w, y+h), color, 2)
            
            # Draw component properties for debugging
            info_text = f"w:{w},h:{h},ar:{h/w:.1f}"
            cv2.putText(debug_binary, info_text, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save debug images
        debug_dir = os.path.join(output_folder, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Extract and save individual characters
        for i, (x, y, w, h, comp_type) in enumerate(valid_components):
            try:
                if i < len(true_chars):
                    true_char = true_chars[i]
                    
                    # Create character folder if it doesn't exist
                    char_folder = os.path.join(output_folder, true_char)
                    if not os.path.exists(char_folder):
                        os.makedirs(char_folder)
                    
                    # Extract and save character image
                    pad = 5
                    x1 = max(0, x - pad)
                    x2 = min(width, x + w + pad)
                    y1 = max(0, y - pad)
                    y2 = min(height, y + h + pad)
                    
                    char_img = gray[y1:y2, x1:x2]
                    char_img = cv2.resize(char_img, (28, 28))
                    
                    save_path = os.path.join(char_folder, f"{filename}_{i}.png")
                    cv2.imwrite(save_path, char_img)
                    
                    # Add label to debug image
                    cv2.putText(debug_original, true_char, (x, y-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
            except Exception as e:
                print(f"Error processing character {i} from {filename}: {str(e)}")
                continue
        
        # Save final debug images
        cv2.imwrite(os.path.join(debug_dir, f'binary_{filename}'), binary)
        cv2.imwrite(os.path.join(debug_dir, f'predict_{filename}'), debug_original)

if __name__ == "__main__":
    extract_digits("raw_images", "extracted_digits") 