import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

def augment_images(input_folder, output_folder, augmentations_per_image=15):
    """Generate augmented versions of existing digit images"""
    
    # Create more focused augmentation for thin characters
    thin_char_transform = transforms.Compose([
        transforms.RandomRotation((-3, 3)),  # Less rotation for thin chars
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),  # Less scaling
            shear=(-3, 3)  # Less shear
        ),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1
        )
    ])
    
    # Regular transform for other digits
    regular_transform = transforms.Compose([
        transforms.RandomRotation((-10, 10)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=(-10, 10)
        ),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3
        ),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5))
    ])

    # Process each digit folder
    for digit in os.listdir(input_folder):
        # Skip the debug folder
        if digit == 'debug':
            continue
            
        digit_path = os.path.join(input_folder, digit)
        if not os.path.isdir(digit_path):
            continue
            
        # Create output folder for this digit
        out_digit_path = os.path.join(output_folder, digit)
        os.makedirs(out_digit_path, exist_ok=True)
        
        # Use different augmentation based on character type
        if digit in ['1', ',']:
            transform = thin_char_transform
            num_aug = augmentations_per_image * 2  # Double augmentations for underrepresented classes
        else:
            transform = regular_transform
            num_aug = augmentations_per_image
        
        # Process each image in the digit folder
        for img_name in os.listdir(digit_path):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Load image
            img_path = os.path.join(digit_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Generate augmented versions
                for i in range(num_aug):
                    # Apply transforms
                    augmented = transform(img)
                    
                    # Save augmented image
                    save_name = f"{os.path.splitext(img_name)[0]}_aug{i}.png"
                    save_path = os.path.join(out_digit_path, save_name)
                    augmented.save(save_path)
                
                # Also copy original image
                img.save(os.path.join(out_digit_path, img_name))
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

def main():
    input_folder = "extracted_digits"
    output_folder = "augmented_digits"
    
    print("Starting image augmentation...")
    augment_images(input_folder, output_folder, augmentations_per_image=15)
    
    # Print statistics
    total_original = 0
    total_augmented = 0
    print("\nAugmented dataset statistics:")
    for digit in os.listdir(output_folder):
        digit_path = os.path.join(output_folder, digit)
        if os.path.isdir(digit_path):
            num_images = len([f for f in os.listdir(digit_path) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Digit {digit}: {num_images} images")
            total_augmented += num_images
            
    print(f"\nTotal images after augmentation: {total_augmented}")

if __name__ == "__main__":
    main() 