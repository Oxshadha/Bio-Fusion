"""
Model Saving and Loading Utilities
Saves models to Google Drive for persistence and easy sharing
"""

import torch
import os
from pathlib import Path

# ============================================================================
# GOOGLE DRIVE SETUP
# ============================================================================

def setup_google_drive():
    """
    Mount Google Drive and create model directory
    Returns: path to model directory
    """
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
    except:
        print("⚠ Not in Google Colab or Drive already mounted")
    
    # Create model directory in Google Drive
    model_dir = '/content/drive/MyDrive/BioFusion_Models'
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"✓ Model directory: {model_dir}")
    return model_dir

def save_model_to_drive(model, model_name, model_dir=None):
    """
    Save model to Google Drive
    
    Args:
        model: PyTorch model
        model_name: Name for the model file (e.g., 'resnet50_final')
        model_dir: Directory to save (default: Google Drive)
    
    Returns:
        Full path to saved model
    """
    if model_dir is None:
        model_dir = setup_google_drive()
    
    # Save model weights
    model_path = os.path.join(model_dir, f'{model_name}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Also save locally (for immediate use)
    local_path = f'{model_name}.pt'
    torch.save(model.state_dict(), local_path)
    print(f"✓ Model also saved locally: {local_path}")
    
    return model_path

def load_model_from_drive(model, model_name, model_dir=None, device='cuda'):
    """
    Load model from Google Drive or local
    
    Args:
        model: PyTorch model instance
        model_name: Name of the model file (e.g., 'resnet50_final')
        model_dir: Directory to load from (default: Google Drive)
        device: Device to load model to
    
    Returns:
        Loaded model
    """
    if model_dir is None:
        # Try Google Drive first
        drive_path = '/content/drive/MyDrive/BioFusion_Models'
        drive_model_path = os.path.join(drive_path, f'{model_name}.pt')
        
        # Try local second
        local_path = f'{model_name}.pt'
        
        if os.path.exists(drive_model_path):
            model_path = drive_model_path
            print(f"✓ Loading from Google Drive: {model_path}")
        elif os.path.exists(local_path):
            model_path = local_path
            print(f"✓ Loading from local: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_name}.pt")
    else:
        model_path = os.path.join(model_dir, f'{model_name}.pt')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")
    
    return model

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def inference_on_image(model, image_path, transform, device, class_names):
    """
    Run inference on a single image
    
    Args:
        model: Trained model
        image_path: Path to image file
        transform: Image preprocessing transform
        device: Device to run on
        class_names: List of class names
    
    Returns:
        prediction, confidence, probabilities
    """
    from PIL import Image
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    prob_dict = {class_names[i]: probabilities[0][i].item() 
                 for i in range(len(class_names))}
    
    return predicted_class, confidence_score, prob_dict

def inference_on_folder(model, folder_path, transform, device, class_names):
    """
    Run inference on all images in a folder
    
    Args:
        model: Trained model
        folder_path: Path to folder with images
        transform: Image preprocessing transform
        device: Device to run on
        class_names: List of class names
    
    Returns:
        List of predictions
    """
    import glob
    from PIL import Image
    
    results = []
    image_files = glob.glob(os.path.join(folder_path, '*.png')) + \
                  glob.glob(os.path.join(folder_path, '*.jpg')) + \
                  glob.glob(os.path.join(folder_path, '*.jpeg'))
    
    model.eval()
    with torch.no_grad():
        for image_path in image_files:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            results.append({
                'image': os.path.basename(image_path),
                'predicted_class': class_names[predicted.item()],
                'confidence': confidence.item(),
                'probabilities': {class_names[i]: probabilities[0][i].item() 
                               for i in range(len(class_names))}
            })
    
    return results

# ============================================================================
# NOTEBOOK-SHARING HELPER
# ============================================================================

def create_inference_notebook_template():
    """
    Creates a simple inference notebook that judges can run
    """
    inference_code = '''
# ============================================================================
# INFERENCE NOTEBOOK FOR JUDGES
# ============================================================================

# Step 1: Setup
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Install dependencies
!pip install torch torchvision timm

# Step 3: Load model (from Google Drive)
import torch
from shared_utilities import *

# Load your best model
# Example: Load ViT-Base
from model_uni import UNIClassifier
model = UNIClassifier(num_classes=8, dropout=0.5)
model = load_model_from_drive(model, 'uni_final', device='cuda')

# Step 4: Prepare transforms
from shared_utilities import get_transforms
transform = get_transforms(augment=False)

# Step 5: Run inference
# On single image
predicted_class, confidence, probs = inference_on_image(
    model, 
    'path/to/image.png',
    transform,
    device='cuda',
    class_names=CLASSES
)

print(f"Predicted: {predicted_class}")
print(f"Confidence: {confidence:.4f}")

# On folder of images
results = inference_on_folder(
    model,
    'path/to/folder',
    transform,
    device='cuda',
    class_names=CLASSES
)

# Print results
for result in results:
    print(f"{result['image']}: {result['predicted_class']} ({result['confidence']:.4f})")
'''
    return inference_code

