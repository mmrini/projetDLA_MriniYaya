"""
Explainability tools for GoogLeNet.

Includes:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Occlusion Sensitivity
- Feature extraction and t-SNE visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2

#==============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Produces heatmaps showing which regions of an image are important for predictions.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients from (e.g., model.inception5b)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image, target_class=None):
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            cam: Grad-CAM heatmap (H, W)
            predicted_class: Predicted class index
        """
        self.model.eval()
        image = image.requires_grad_(True)
        
        # Forward pass
        output = self.model(image)
        if isinstance(output, tuple):
            output = output[0]
        
        # Get predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=image.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), target_class
    
    def visualize(self, image, cam, original_size=(224, 224), alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Create visualization by overlaying CAM on original image.
        
        Args:
            image: Original image tensor (C, H, W) or numpy array
            cam: CAM heatmap (H, W)
            original_size: Size to resize visualization
            alpha: Blending factor
            colormap: OpenCV colormap
        
        Returns:
            visualization: Blended image with heatmap
        """
        # Convert image to numpy if needed
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                image = np.transpose(image, (1, 2, 0))
        
        # Denormalize if needed (assuming ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, original_size)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255
        
        # Blend
        visualization = alpha * heatmap + (1 - alpha) * image
        visualization = np.clip(visualization, 0, 1)
        
        return visualization


#==============================================================================

class OcclusionSensitivity:
    """
    Occlusion-based sensitivity analysis.
    
    Systematically occludes parts of the image to see effect on prediction.
    """
    
    def __init__(self, model, occlusion_size=32, occlusion_stride=16):
        """
        Args:
            model: PyTorch model
            occlusion_size: Size of occlusion patch
            occlusion_stride: Stride for sliding window
        """
        self.model = model
        self.occlusion_size = occlusion_size
        self.occlusion_stride = occlusion_stride
        self.model.eval()
    
    def generate_heatmap(self, image, target_class=None, occlusion_value=0.5):
        """
        Generate occlusion sensitivity heatmap.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class (if None, use predicted)
            occlusion_value: Value to use for occlusion
        
        Returns:
            heatmap: Sensitivity heatmap (H, W)
            predicted_class: Predicted class
        """
        batch_size, channels, height, width = image.shape
        device = image.device
        
        # Get baseline prediction
        with torch.no_grad():
            output = self.model(image)
            if isinstance(output, tuple):
                output = output[0]
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            baseline_score = F.softmax(output, dim=1)[0, target_class].item()
        
        # Create heatmap
        heatmap = np.zeros((height, width))
        count_map = np.zeros((height, width))
        
        # Slide occlusion window
        for y in range(0, height - self.occlusion_size + 1, self.occlusion_stride):
            for x in range(0, width - self.occlusion_size + 1, self.occlusion_stride):
                # Create occluded image
                occluded = image.clone()
                occluded[:, :, y:y+self.occlusion_size, x:x+self.occlusion_size] = occlusion_value
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(occluded)
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    score = F.softmax(output, dim=1)[0, target_class].item()
                
                # Record drop in confidence
                sensitivity = baseline_score - score
                heatmap[y:y+self.occlusion_size, x:x+self.occlusion_size] += sensitivity
                count_map[y:y+self.occlusion_size, x:x+self.occlusion_size] += 1
        
        # Average overlapping regions
        heatmap = heatmap / np.maximum(count_map, 1)
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap, target_class

#==============================================================================

def extract_features(model, dataloader, layer_name, device='cuda', max_batches=None):
    """
    Extract features from a specific layer of the model.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        layer_name: Name of layer to extract from (e.g., 'inception5b')
        device: Device to use
        max_batches: Maximum number of batches (None = all)
    
    Returns:
        features: Extracted features (N, D)
        labels: Corresponding labels (N,)
    """
    model.eval()
    
    # Get the target layer
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    # Hook to capture features
    features_list = []
    
    def hook_fn(module, input, output):
        # Global average pooling
        if len(output.shape) == 4:  # (B, C, H, W)
            pooled = F.adaptive_avg_pool2d(output, (1, 1))
            features_list.append(pooled.squeeze().detach().cpu())
        else:
            features_list.append(output.detach().cpu())
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    labels_list = []
    batch_count = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            if max_batches and batch_count >= max_batches:
                break
            
            images = images.to(device)
            _ = model(images)
            labels_list.append(labels)
            
            batch_count += 1
    
    handle.remove()
    
    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    
    return features, labels


def compute_tsne(features, labels, n_components=2, perplexity=30, n_iter=1000):
    """
    Compute t-SNE embedding of features.
    
    Args:
        features: Feature vectors (N, D)
        labels: Labels (N,)
        n_components: Dimensionality of embedding
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
    
    Returns:
        embedding: t-SNE embedding (N, n_components)
    """
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=n_iter, random_state=42, verbose=1)
    embedding = tsne.fit_transform(features)
    return embedding


def visualize_tsne(embedding, labels, class_names=None, save_path=None):
    """
    Visualize t-SNE embedding.
    
    Args:
        embedding: t-SNE embedding (N, 2)
        labels: Labels (N,)
        class_names: List of class names
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        label_name = class_names[label] if class_names else f"Class {label}"
        plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                   c=[colors[i]], label=label_name, alpha=0.6, s=20)
    
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('t-SNE Visualization of Learned Features', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

