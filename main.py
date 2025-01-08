import cv2
import os
from predict import extract_and_predict
import torch
from train import create_model

def cut_image(input_path, output_dir="cropped_images", target_size=(224, 224)):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Could not read image: {input_path}")
        return
    
    # Define the coordinates for each cut [x, y, width, height]
    cuts = [
        [50, 75, 225, 50],    # Cut 1
        [275, 75, 235, 50],   # Cut 2
        [510, 75, 240, 50],   # Cut 3
        [50, 145, 225, 45],   # Cut 4
        [275, 145, 235, 45],  # Cut 5
        [510, 145, 240, 45],  # Cut 6
        [50, 215, 225, 40],   # Cut 7
        [275, 215, 235, 40],  # Cut 8
        [510, 215, 240, 40],  # Cut 9
        [50, 280, 225, 40],   # Cut 10
        [275, 280, 235, 40],  # Cut 11
        [510, 280, 240, 40]   # Cut 12
    ]
    
    cut_paths = []  # Store paths of cropped images
    
    # Process each cut
    for i, (x, y, w, h) in enumerate(cuts, 1):
        # Extract the region
        cropped = img[y:y+h, x:x+w]
        
        # Resize the cropped image
        cropped_resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Save the cropped image
        output_path = os.path.join(output_dir, f"cut_{i}.png")
        cv2.imwrite(output_path, cropped_resized)
        cut_paths.append(output_path)
        print(f"Saved cut {i} to {output_path}")
        
        # Draw rectangle on original image to verify cuts
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save debug image with rectangles
    cv2.imwrite(os.path.join(output_dir, "debug_cuts.png"), img)
    print("\nAll cuts completed. Check the 'cropped_images' folder.")
    
    return cut_paths

def process_image(image_path):
    # Cut the image
    print(f"\nProcessing: {image_path}")
    cut_paths = cut_image(image_path)
    
    # Set up model for prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_model()
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Process each cut
    print("\nPredicting numbers for each cut:")
    results = []
    for i, cut_path in enumerate(cut_paths, 1):
        result = extract_and_predict(model, cut_path, device)
        results.append({
            "question": i,
            "answer": result
        })
        print(f"Cut {i}: {result}")
    
    # Save results to JSON file
    import json
    output_filename = os.path.splitext(image_path)[0] + "_results.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_filename}")

def main():
    # Process each image in the current directory
    for filename in os.listdir():
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            process_image(filename)

if __name__ == "__main__":
    main()