"""
Adversarial attack implementations for GoogLeNet evaluation.

Includes:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
"""

import torch
import torch.nn as nn

#==============================================================================

def fgsm_attack(model, images, labels, epsilon, loss_fn=None, targeted=False, target_labels=None):
    """
    Fast Gradient Sign Method (FGSM) attack.
        
    Args:
        model: PyTorch model (will be set to eval mode)
        images: Input images tensor (B, C, H, W)
        labels: True labels (B,)
        epsilon: Perturbation magnitude (e.g., 8/255)
        loss_fn: Loss function (default: CrossEntropyLoss)
        targeted: If True, perform targeted attack
        target_labels: Target labels for targeted attack (B,)
    
    Returns:
        adv_images: Adversarial images (B, C, H, W)
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    model.eval()
    
    # Clone images and enable gradient tracking
    images = images.clone().detach().to(images.device)
    labels = labels.clone().detach().to(images.device)
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    
    # Handle auxiliary classifiers output during training mode
    if isinstance(outputs, tuple):
        outputs = outputs[0]  # Use main output
    
    # Calculate loss
    if targeted and target_labels is not None:
        loss = -loss_fn(outputs, target_labels)  # Maximize loss for target
    else:
        loss = loss_fn(outputs, labels)  # Maximize loss for true label
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Get sign of gradient
    data_grad = images.grad.data
    
    # Create adversarial examples
    sign_data_grad = data_grad.sign()
    perturbed_images = images + epsilon * sign_data_grad
    
    return perturbed_images.detach()


#==============================================================================

def pgd_attack(model, images, labels, epsilon, alpha, num_steps, 
               loss_fn=None, targeted=False, target_labels=None, random_start=True):
    """
    Projected Gradient Descent (PGD) attack.
        
    Args:
        model: PyTorch model
        images: Input images tensor (B, C, H, W)
        labels: True labels (B,)
        epsilon: Maximum perturbation (L_inf bound)
        alpha: Step size per iteration
        num_steps: Number of attack iterations
        loss_fn: Loss function (default: CrossEntropyLoss)
        targeted: If True, perform targeted attack
        target_labels: Target labels for targeted attack
        random_start: If True, start from random point in epsilon ball
    
    Returns:
        adv_images: Adversarial images (B, C, H, W)
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    model.eval()
    
    images = images.clone().detach().to(images.device)
    labels = labels.clone().detach().to(images.device)
    
    # Random initialization
    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        #adv_images = torch.clamp(adv_images, 0, 1)
    else:
        adv_images = images.clone()
    
    # Iterative attack
    for _ in range(num_steps):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Handle auxiliary classifiers
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # Calculate loss
        if targeted and target_labels is not None:
            loss = -loss_fn(outputs, target_labels)
        else:
            loss = loss_fn(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get gradient
        data_grad = adv_images.grad.data
        
        # Update adversarial images
        with torch.no_grad():
            adv_images = adv_images + alpha * data_grad.sign()
            
            # Project back to epsilon ball around original image
            perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = images + perturbation
    
    return adv_images.detach()

#==============================================================================

def evaluate_robustness(model, dataloader, attack_fn, device, max_batches=None):
    """
    Evaluate model robustness against an attack.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        attack_fn: Attack function that takes (model, images, labels) and returns adv_images
        device: Device to use
        max_batches: Maximum number of batches to evaluate (None = all)
    
    Returns:
        dict with metrics:
            - clean_accuracy: Accuracy on clean images
            - robust_accuracy: Accuracy on adversarial images
            - attack_success_rate: Percentage of successful attacks
    """
    model.eval()
    
    clean_correct = 0
    robust_correct = 0
    total = 0
    
    batch_count = 0
    
    for images, labels in dataloader:
        if max_batches and batch_count >= max_batches:
            break
        
        images, labels = images.to(device), labels.to(device)
        
        # Clean accuracy (no grad needed for forward pass)
        with torch.no_grad():
            outputs_clean = model(images)
            if isinstance(outputs_clean, tuple):
                outputs_clean = outputs_clean[0]
            pred_clean = outputs_clean.argmax(1)
            clean_correct += (pred_clean == labels).sum().item()
        
        # Generate adversarial examples (needs gradients)
        adv_images = attack_fn(model, images, labels)
        
        # Robust accuracy (no grad needed for forward pass)
        with torch.no_grad():
            outputs_adv = model(adv_images)
            if isinstance(outputs_adv, tuple):
                outputs_adv = outputs_adv[0]
            pred_adv = outputs_adv.argmax(1)
            robust_correct += (pred_adv == labels).sum().item()
        
        total += labels.size(0)
        batch_count += 1
    
    clean_acc = 100.0 * clean_correct / total
    robust_acc = 100.0 * robust_correct / total
    attack_success_rate = 100.0 * (clean_correct - robust_correct) / clean_correct if clean_correct > 0 else 0
    
    return {
        'clean_accuracy': clean_acc,
        'robust_accuracy': robust_acc,
        'attack_success_rate': attack_success_rate,
        'total_samples': total
    }

