import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import joblib # Import joblib for saving

class ImageProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=5) # Ensure this matches the previous fix
        
    def create_sample_images(self):
        """
        Create sample facial images for demonstration.
        Skips generation if real images are already present in a member's directory.
        """
        os.makedirs('data/images', exist_ok=True)
        
        # Define 6 members and the specified emotions
        members = ['member1', 'member2', 'member3', 'member4', 'member5', 'member6']
        emotions = ['neutral', 'smiling', 'normal']
        
        for member in members:
            member_dir = f'data/images/{member}'
            os.makedirs(member_dir, exist_ok=True)
            
            # Check if the directory already contains .jpeg or .png files
            existing_images = [f for f in os.listdir(member_dir) if f.endswith(('.jpeg', '.png', '.jpg'))]
            
            if existing_images:
                print(f"Skipping synthetic image generation for {member}: Real images already found.")
                continue # Skip to the next member if real images are present
            
            # If no real images, generate synthetic ones
            image_counter = 0
            for emotion in emotions:
                # Create a simple colored rectangle as placeholder
                img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                
                # Add some pattern to make it look different based on emotion
                if emotion == 'neutral':
                    img[:, :, 0] = 100  # More blue
                elif emotion == 'smiling':
                    img[:, :, 1] = 150  # More green
                else:  # 'normal'
                    img[:, :, 2] = 200  # More red
                
                # Save with a unique name for each emotion type per member, using .jpeg
                cv2.imwrite(f'{member_dir}/{emotion}_{image_counter}.jpeg', img)
                image_counter += 1
       
        print("Sample images created successfully for 6 members (or existing real images preserved)!")
   
    def load_and_display_images(self):
        """
        Load and display sample images
        """
        members_dirs = [d for d in os.listdir('data/images') if os.path.isdir(os.path.join('data/images', d))]
        
        if not members_dirs:
            print("No image directories found to display.")
            return

        # Determine the number of images per member to display (e.g., 3: neutral, smiling, normal)
        # We'll try to display one of each type if available
        emotions_to_display = ['neutral', 'smiling', 'normal']
        
        fig, axes = plt.subplots(len(members_dirs), len(emotions_to_display), figsize=(15, len(members_dirs) * 3))
        axes = np.atleast_2d(axes) # Ensure axes is always 2D for consistent indexing
        
        for i, member in enumerate(members_dirs):
            member_dir_path = f'data/images/{member}'
            
            for j, emotion_type in enumerate(emotions_to_display):
                # Find an image that matches the emotion type
                found_img_path = None
                for img_name in os.listdir(member_dir_path):
                    # Check for both .jpg and .jpeg
                    if img_name.startswith(emotion_type) and img_name.endswith(('.jpg', '.jpeg', '.png')):
                        found_img_path = os.path.join(member_dir_path, img_name)
                        break
                
                if found_img_path and os.path.exists(found_img_path):
                    img = cv2.imread(found_img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[i, j].imshow(img_rgb)
                        axes[i, j].set_title(f'{member} - {emotion_type}')
                        axes[i, j].axis('off')
                else:
                    axes[i, j].set_title(f'{member} - {emotion_type} (N/A)')
                    axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('data/sample_images_display.png')
        plt.close()
        
    def apply_augmentations(self, image):
        """
        Apply various augmentations to images
        """
        augmented_images = []
        
        # Original image
        augmented_images.append(('original', image))
        
        # Rotation
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
        rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        augmented_images.append(('rotated', rotated))
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented_images.append(('flipped', flipped))
        
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        augmented_images.append(('grayscale', gray_3channel))
        
        # Brightness adjustment
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_image)
        bright_image = enhancer.enhance(1.3)
        bright_array = cv2.cvtColor(np.array(bright_image), cv2.COLOR_RGB2BGR)
        augmented_images.append(('bright', bright_array))
        
        return augmented_images
    
    def extract_features(self, image):
        """
        Extract features from image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        resized = cv2.resize(gray, (64, 64))
        
        # Flatten image
        flattened = resized.flatten()
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # Calculate texture features (Local Binary Pattern approximation)
        texture_features = []
        # Limiting loop range for texture_features to avoid empty list for small images
        for i in range(1, min(gray.shape[0]-1, 10)): # Limiting to first 10 rows
            for j in range(1, min(gray.shape[1]-1, 10)): # Limiting to first 10 columns
                center = gray[i, j]
                pattern = 0
                pattern += (gray[i-1, j-1] >= center) << 7
                pattern += (gray[i-1, j] >= center) << 6
                pattern += (gray[i-1, j+1] >= center) << 5
                pattern += (gray[i, j+1] >= center) << 4
                pattern += (gray[i+1, j+1] >= center) << 3
                pattern += (gray[i+1, j] >= center) << 2
                pattern += (gray[i+1, j-1] >= center) << 1
                pattern += (gray[i, j-1] >= center) << 0
                texture_features.append(pattern)
        
        # Ensure texture_features is not empty before creating histogram
        if not texture_features:
            texture_hist = np.zeros(256) # Return an array of zeros if no texture features
        else:
            texture_hist = np.histogram(texture_features, bins=256, range=(0, 256))[0]
        
        # Combine features
        features = np.concatenate([
            flattened[:100],  # First 100 pixel values
            hist[::4],        # Every 4th histogram bin
            texture_hist[::4] # Every 4th texture histogram bin
        ])
        
        return features
    
    def process_all_images(self):
        """
        Process all images and extract features
        """
        # Dynamically get members from data/images subdirectories
        members = [d for d in os.listdir('data/images') if os.path.isdir(os.path.join('data/images', d))]
        
        all_features = []
        labels = []
        
        for member in members:
            member_dir_path = f'data/images/{member}'
            if not os.path.exists(member_dir_path):
                continue
            
            for img_name in os.listdir(member_dir_path):
                # Check for .jpg, .jpeg, and .png files
                if img_name.endswith(('.jpg', '.jpeg', '.png')): 
                    img_path = os.path.join(member_dir_path, img_name)
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        print(f"Warning: Could not load image {img_path}")
                        continue
                    
                    # Apply augmentations
                    augmented_images = self.apply_augmentations(image)
                    
                    # Extract features from each augmented image
                    for aug_name, aug_image in augmented_images:
                        features = self.extract_features(aug_image)
                        all_features.append(features)
                        
                        # Create a more robust label based on member, original filename part, and augmentation
                        original_name_part = img_name.split('.')[0] # e.g., 'neutral_0' or 'Diana_neutral'
                        label = f"{member}_{original_name_part}_{aug_name}"
                        labels.append(label)
        
        # Handle cases where no images were processed
        if not all_features:
            print("No image features could be extracted. Check data/images directory.")
            return pd.DataFrame() # Return empty DataFrame
            
        # Convert to numpy array
        features_array = np.array(all_features)
        
        # Normalize features
        self.scaler.fit(features_array) # Fit scaler here
        features_normalized = self.scaler.transform(features_array)
        
        # Apply PCA for dimensionality reduction
        # Check if features_normalized has enough samples for PCA
        if features_normalized.shape[0] < self.pca.n_components:
            print(f"Warning: Not enough samples ({features_normalized.shape[0]}) for PCA n_components={self.pca.n_components}. Adjusting n_components.")
            self.pca.n_components = min(features_normalized.shape[0], features_normalized.shape[1], 1) # Set to a safe minimum
            if self.pca.n_components == 0:
                print("Error: Cannot perform PCA with zero components. Check feature extraction logic.")
                return pd.DataFrame()
        
        self.pca.fit(features_normalized) # Fit PCA here
        
        # Save the fitted scaler and PCA object
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/image_scaler.pkl')
        joblib.dump(self.pca, 'models/image_pca.pkl')
        print("✓ Fitted image scaler and PCA saved to models/image_scaler.pkl and models/image_pca.pkl")

        features_pca = self.pca.transform(features_normalized)

        # Create DataFrame
        feature_columns = [f'feature_{i}' for i in range(features_pca.shape[1])]
        df = pd.DataFrame(features_pca, columns=feature_columns)
        df['label'] = labels
        # Extract member and augmentation from the new label format
        df['member'] = [label.split('_')[0] for label in labels]
        # Emotion might be mixed with filename, simplify for now or refine if needed
        df['emotion'] = ['_'.join(label.split('_')[1:-1]) for label in labels] # Captures part before last underscore
        df['augmentation'] = [label.split('_')[-1] for label in labels]
        
        # Save features
        df.to_csv('data/image_features.csv', index=False)
        
        print(f"Image features extracted and saved!")
        print(f"Features shape: {features_pca.shape}")
        print(f"Number of samples: {len(labels)}")
        
        return df

def main():
    """
    Main function to process images
    """
    processor = ImageProcessor()
    
    print("Creating sample images...")
    processor.create_sample_images()
    
    print("Loading and displaying images...")
    processor.load_and_display_images()
    
    print("Processing images and extracting features...")
    features_df = processor.process_all_images()
    
    print("Image processing complete!")

if __name__ == "__main__":
    main()
