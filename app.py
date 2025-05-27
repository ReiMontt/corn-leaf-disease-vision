import streamlit as st
import torch
import tensorflow as tf
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import timm
import torch.nn.functional as F
import os
import math
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.metrics import F1Score
import gdown # Import gdown

# --- Constants ---
IMG_SIZE = (224, 224)
PATCH_SIZE = 16
VIT_MODEL_PATH = "./vit_corn_model.pth"
CNN_MODEL_PATH = "./cnn_corn_model.h5"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DATA_DIR = "./corn_data/train" # Still used for dynamic class names, even if data isn't downloaded
NUM_PATCHES = (IMG_SIZE[0] // PATCH_SIZE) ** 2

# Google Drive IDs for model files
VIT_MODEL_GDRIVE_ID = "1_Rm1-AnxrRXaSmCGRYafWVQZGx79rXBD"
CNN_MODEL_GDRIVE_ID = "1XFLVxaT222PHNKkHONS7zzVoVxKOu4ML"

# Dynamically load class names
def get_class_names(train_dir=TRAIN_DATA_DIR):
    """
    Attempts to load class names from the training data directory.
    Falls back to a default list if the directory is not found.
    """
    if os.path.exists(train_dir) and os.listdir(train_dir):
        return sorted(os.listdir(train_dir))
    else:
        # Default class names if training data directory is not available
        return ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

CLASS_NAMES = get_class_names()

# --- Model Loading ---
@st.cache_resource
def load_model(model_type, num_classes=len(CLASS_NAMES)):
    """
    Loads the specified deep learning model (ViT or CNN).
    Downloads the model file from Google Drive if not found locally.
    Uses caching for efficiency.
    """
    if model_type == "ViT":
        if not os.path.exists(VIT_MODEL_PATH):
            st.info(f"ViT model not found locally. Downloading from Google Drive...")
            try:
                gdown.download(f"https://drive.google.com/uc?id={VIT_MODEL_GDRIVE_ID}", VIT_MODEL_PATH, quiet=False)
                st.success("ViT model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download ViT model: {str(e)}")
        
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        try:
            model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=DEVICE))
        except Exception as e:
            raise RuntimeError(f"Failed to load ViT model weights: {str(e)}")
        model.to(DEVICE)
        model.eval()

        # Register hooks for all ViT transformer blocks to capture attention weights
        attention_weights = [[] for _ in range(len(model.blocks))]
        def get_hook_fn(block_idx):
            def hook_fn(module, input, output):
                try:
                    x = input[0]
                    B, N, C = x.shape
                    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    attn = (q @ k.transpose(-2, -1)) * module.scale
                    attn = attn.softmax(dim=-1)
                    attention_weights[block_idx].append(attn.detach())
                except Exception as e:
                    # Fallback for attention weights if capture fails
                    dummy_attn = torch.ones((B, module.num_heads, N, N), device=x.device) / N
                    attention_weights[block_idx].append(dummy_attn)
            return hook_fn

        for i in range(len(model.blocks)):
            model.blocks[i].attn.register_forward_hook(get_hook_fn(i))
        
        return model, attention_weights, "ViT"
    
    elif model_type == "CNN":
        if not os.path.exists(CNN_MODEL_PATH):
            st.info(f"CNN model not found locally. Downloading from Google Drive...")
            try:
                gdown.download(f"https://drive.google.com/uc?id={CNN_MODEL_GDRIVE_ID}", CNN_MODEL_PATH, quiet=False)
                st.success("CNN model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download CNN model: {str(e)}")
        
        try:
            model = tf.keras.models.load_model(CNN_MODEL_PATH)

            # Ensure the model's graph is fully built for consistent behavior
            dummy_input = tf.zeros((1, *IMG_SIZE, 3))
            _ = model(dummy_input)

            # Create a separate VGG16 base for feature map extraction
            vgg16_feature_extractor = VGG16(weights=None, include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
            
            # Transfer weights from the loaded model's VGG16 base layer
            # Your training script places VGG16 at model.layers[0]
            loaded_vgg16_weights = model.layers[0].get_weights()
            vgg16_feature_extractor.set_weights(loaded_vgg16_weights)
            
            # Compile the feature extractor (optional, but ensures graph is ready)
            vgg16_feature_extractor.compile(optimizer='adam', loss='mse')
            
            feature_model = vgg16_feature_extractor
            
            return model, feature_model, "CNN"

        except Exception as e:
            raise RuntimeError(f"Failed to load and prepare CNN model: {str(e)}")

# --- Image Preprocessing ---
def preprocess_image_vit(image):
    """Preprocesses image for ViT model (PyTorch)."""
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    return image_tensor

def preprocess_image_cnn(image):
    """Preprocesses image for CNN model (TensorFlow)."""
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype('float32')
    image_array = np.expand_dims(image_array, axis=0)
    
    # Apply VGG16-specific preprocessing for consistency with training
    image_array = tf.keras.applications.vgg16.preprocess_input(image_array)
    
    return image_array

# --- Visualization Functions ---
def visualize_patches(image):
    """Visualizes how the input image is divided into patches for ViT."""
    image = image.convert('RGB').resize(IMG_SIZE)
    image_np = np.array(image)
    patches = []
    for i in range(0, IMG_SIZE[0], PATCH_SIZE):
        for j in range(0, IMG_SIZE[1], PATCH_SIZE):
            patch = image_np[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            patches.append(patch)
    
    # Assuming 14x14 patches for 224x224 image with 16x16 patch size
    fig, ax = plt.subplots(14, 14, figsize=(8, 8))
    for idx, patch in enumerate(patches):
        row, col = idx // 14, idx % 14
        ax[row, col].imshow(patch)
        ax[row, col].axis('off')
    plt.tight_layout()
    return fig

def visualize_attention_map(attention_weights, image_shape=(224, 224), image=None):
    """
    Visualizes attention maps from ViT transformer blocks, overlaid on the input image.
    If no attention weights are captured, uniform maps are used.
    """
    if not any(attention_weights) or all(not block_weights for block_weights in attention_weights):
        attn_maps = [np.ones((14, 14)) / 196 for _ in range(12)] # Default for 12 blocks, 14x14 patches
    else:
        attn_maps = []
        for block_idx, block_weights in enumerate(attention_weights):
            if not block_weights or block_weights[-1] is None:
                attn_maps.append(np.ones((14, 14)) / 196) # Uniform if specific block fails
                continue
            
            try:
                attn = block_weights[-1].mean(dim=1)[0]  # [num_patches+1, num_patches+1]
                if attn.dim() != 2 or attn.shape[0] < 2:
                    attn_maps.append(np.ones((14, 14)) / 196) # Uniform if shape is incorrect
                    continue
                attn = attn[1:, 1:].mean(dim=0)[:196].reshape(14, 14) # Exclude CLS token, reshape
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8) # Normalize to [0, 1]
                attn_maps.append(attn.cpu().numpy())
            except Exception:
                attn_maps.append(np.ones((14, 14)) / 196) # Uniform on error

    fig, axes = plt.subplots(4, 3, figsize=(12, 16)) # Assuming 12 transformer blocks
    axes = axes.flatten()
    image_np = np.array(image.resize(image_shape))
    
    for idx, (attn, ax) in enumerate(zip(attn_maps, axes)):
        # Resize attention map to image size for overlay
        attn_resized = F.interpolate(
            torch.tensor(attn).unsqueeze(0).unsqueeze(0),
            size=image_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        ax.imshow(image_np)
        ax.imshow(attn_resized, cmap='jet', alpha=0.5) # Overlay with transparency
        ax.axis('off')
        ax.set_title(f'Block {idx+1} Attention')
    
    plt.tight_layout()
    return fig

def visualize_feature_maps(feature_model, preprocessed_image_array, num_maps_to_show=16):
    """
    Visualizes feature maps from a CNN model's output layer.
    """
    if feature_model is None or len(feature_model.output.shape) != 4:
        st.error("Feature maps are not in the expected 4D format. Visualization will be skipped.")
        return None

    try:
        feature_maps = feature_model.predict(preprocessed_image_array, verbose=0) # Set verbose to 0 to suppress Keras output
        feature_maps = np.squeeze(feature_maps, axis=0) # Remove the batch dimension

        total_channels = feature_maps.shape[-1]
        
        if num_maps_to_show is None or num_maps_to_show > total_channels:
            num_maps_to_show = total_channels
        
        display_maps = feature_maps[:, :, :num_maps_to_show]

        cols = 4
        rows = math.ceil(num_maps_to_show / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()

        for i in range(num_maps_to_show):
            ax = axes[i]
            # Plot the 2D slice of the feature map
            ax.imshow(display_maps[:, :, i], cmap='viridis')
            ax.set_title(f"Feature Map {i+1}")
            ax.axis('off')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"An error occurred while generating feature maps: {e}. Visualization will be skipped.")
        return None

# --- Prediction Functions ---
def predict_vit(image_tensor, model):
    """Performs prediction using the ViT model."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx] * 100
    return predicted_class, confidence, probabilities

def predict_cnn(image_array, model):
    """Performs prediction using the CNN model."""
    try:
        outputs = model.predict(image_array, verbose=0)[0]
        probabilities = tf.nn.softmax(outputs).numpy()
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = probabilities[predicted_class_idx] * 100
    except Exception as e:
        st.error(f"Error during CNN prediction: {str(e)}")
        return None, None, None
    return predicted_class, confidence, probabilities

# --- Probability Visualization ---
def plot_probabilities(probabilities, class_names):
    """Visualizes the model's class probability distribution."""
    probabilities = np.array(probabilities)
    paired = list(zip(probabilities, class_names))
    paired_sorted = sorted(paired, key=lambda x: x[0], reverse=True)
    sorted_probs, sorted_names = zip(*paired_sorted)
    sorted_probs = np.array(sorted_probs)
    sorted_names = list(sorted_names)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    y_pos = np.arange(len(sorted_names))[::-1]
    bars = ax.barh(y_pos, sorted_probs * 100, color='skyblue', height=0.4)
    
    for bar in bars:
        width = bar.get_width().item() if isinstance(bar.get_width(), np.ndarray) else bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%', 
                va='center', ha='left', fontsize=10)
    
    bars[0].set_color('orange') # Highlight the highest probability
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Class Probability Distribution')
    ax.set_xlim(0, 110)
    
    plt.tight_layout()
    return fig

# --- Main Streamlit App ---
def main():
    st.set_page_config(layout="wide") # Use wide layout for better visualization space
    st.title("Corn Disease Classification and Model Insights")
    st.write("Upload a corn leaf image to classify diseases and visualize how the model processes the input.")

    # Model selection
    model_type = st.sidebar.selectbox("Select Model for Analysis", ["ViT", "CNN"])

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload a corn leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Step 1: Display uploaded image
        st.header("1. Input Image")
        st.write("The uploaded corn leaf image is resized to 224x224 pixels for model processing.")
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception:
            st.error("Invalid image file. Please upload a valid JPG or PNG image.")
            return
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption='Original Uploaded Image', width=250)
        with col2:
            st.image(image.resize(IMG_SIZE), caption=f'Resized to {IMG_SIZE[0]}x{IMG_SIZE[1]} pixels', width=250)


        # Load model
        st.header("2. Model Loading")
        st.write(f"Loading the selected {model_type} model. This happens once and is cached for speed.")
        try:
            model, aux_data, loaded_type = load_model(model_type)
        except (FileNotFoundError, RuntimeError) as e:
            st.error(str(e))
            return

        if model_type == "ViT":
            # Step 3: Patch Embedding
            st.header("3. Vision Transformer Processing")
            st.subheader("3.1 Patch Embedding")
            st.write("The ViT first divides the image into 16x16 pixel patches. Each patch is then converted into a numerical embedding.")
            patch_fig = visualize_patches(image)
            st.pyplot(patch_fig)

            # Prediction
            image_tensor = preprocess_image_vit(image)
            predicted_class, confidence, probabilities = predict_vit(image_tensor, model)

            # Step 4: Transformer Blocks
            st.subheader("3.2 Transformer Blocks & Attention Mechanism")
            st.write("These layers process the patch embeddings, using self-attention to weigh the importance of different image regions for classification. The attention maps below highlight areas the model focused on.")
            attn_fig = visualize_attention_map(aux_data, IMG_SIZE, image)
            if attn_fig:
                st.pyplot(attn_fig)
            else:
                st.warning("Attention map visualization is not available for this model.")

        elif model_type == "CNN":
            # Step 3: Feature Extraction (CNN)
            st.header("3. Convolutional Neural Network Processing")
            st.subheader("3.1 Feature Map Extraction")
            st.write("The CNN extracts hierarchical features through its convolutional layers. Early layers detect basic patterns like edges and textures, while deeper layers identify more complex features relevant to the disease.")
            
            image_array = preprocess_image_cnn(image)
            feature_fig = visualize_feature_maps(aux_data, image_array, num_maps_to_show=32) # Display more maps
            if feature_fig:
                st.pyplot(feature_fig)
            else:
                st.warning("Feature map visualization is not available for this model.")

            # Prediction
            predicted_class, confidence, probabilities = predict_cnn(image_array, model)
            if predicted_class is None:
                return

        # Step 4: Classification
        st.header("4. Classification Result")
        st.write("The model's final output is a probability distribution across all possible disease classes.")
        if probabilities is not None:
            st.success(f"**Predicted Class**: **{predicted_class}** with **{confidence:.2f}%** confidence.")
            prob_fig = plot_probabilities(probabilities, CLASS_NAMES)
            st.pyplot(prob_fig)
        else:
            st.error("Prediction failed. Please try again.")

if __name__ == "__main__":
    main()