"""
Fine-tuning script for pretrained GoogLeNet (Inception V1) on CIFAR-10

"""

import torch
import torch.nn as nn
from torchvision import transforms 
import torchvision.models as models
from PIL import Image

#==============================================================================

def load_pretrained_googlenet(num_classes, freeze_backbone=True, use_aux_logits=False): # If fine-tuning on a dataset as small as CIFAR-10, to avoid overfitting
    """
    Load pretrained GoogLeNet and adapt for target dataset.
    
    Args:
        num_classes: Number of classes in target dataset
        freeze_backbone: If True, freeze all layers except final classifier
        use_aux_logits: If False, disable auxiliary classifiers (recommended for fine-tuning)
    
    Returns:
        Modified GoogLeNet model
    """

    # Load pretrained model
    # For fine-tuning, we disable aux_logits since:
    # 1. Auxiliary classifiers have random weights (not pretrained for the new task)
    # 2. They add noise to gradient signal when training only the main classifier
    # 3. They slow down training with unnecessary computation
    try:
        model = models.googlenet(weights='IMAGENET1K_V1', aux_logits=use_aux_logits) # for torchvision >= 0.13
    except TypeError:
        model = models.googlenet(pretrained=True, aux_logits=use_aux_logits) # for older torchvision versions
    
    print("Pretrained weights loaded from ImageNet")
    
    # Replace final FC layer for new number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Update auxiliary classifiers if present and enabled
    if use_aux_logits:
        if hasattr(model, 'aux1') and model.aux1 is not None:
            num_ftrs_aux = model.aux1.fc2.in_features
            model.aux1.fc2 = nn.Linear(num_ftrs_aux, num_classes)
        
        if hasattr(model, 'aux2') and model.aux2 is not None:
            num_ftrs_aux = model.aux2.fc2.in_features
            model.aux2.fc2 = nn.Linear(num_ftrs_aux, num_classes)
    
    print(f"Final layer adapted for {num_classes} classes")
    
    # Freeze backbone
    if freeze_backbone:
        print("Freezing backbone layers (only training final classifier)...")
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Don't freeze fc layers
                param.requires_grad = False
        print("Backbone frozen")
    
    return model

#==============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: GoogLeNet model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs) # (main_output, aux1, aux2) 
        
        # When aux_logits=False, outputs is just the main classifier
        # When aux_logits=True, outputs is (main_output, aux1, aux2)

        if isinstance(outputs, tuple):
            main_output, aux1, aux2 = outputs
            loss_main = criterion(main_output, targets)
            loss_aux1 = criterion(aux1, targets)
            loss_aux2 = criterion(aux2, targets)
            # Combined loss with auxiliary classifiers weighted at 0.3 (as in paper)
            loss = loss_main + 0.3 * (loss_aux1 + loss_aux2)
            outputs = main_output  # Use main output for accuracy
        else:
            # Standard case for fine-tuning: only main classifier output
            loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(dataloader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


#==============================================================================

def multi_crop_evaluate(model, dataset, criterion, device, num_crops=1, crop_size=224, batch_size=32):
    """
    Evaluate model using multi-crop testing as in GoogLeNet paper.
    
    The paper uses:
    - 1 crop: center crop only
    - 10 crops: 4 corners + center, each with horizontal flip
    - 144 crops: dense sampling at multiple scales
    
    Args:
        model: GoogLeNet model
        dataset: Dataset to evaluate on
        device: Device to use
        num_crops: Number of crops (1, 10, or 144)
        crop_size: Size of crops (224 for GoogLeNet)
        batch_size: Batch size for processing
        criterion: Loss function (optional, for loss calculation)
    
    Returns:
        Dictionary containing:
            'top1_acc': Top-1 accuracy (%)
            'top5_acc': Top-5 accuracy (%)
            'top1_err': Top-1 error rate (%) - for comparison with paper
            'top5_err': Top-5 error rate (%) - for comparison with paper
            'loss': Average loss (if criterion provided, otherwise None)
    """
    model.eval()
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    running_loss = 0.0
    
    print(f"\nEvaluating with {num_crops}-crop testing...")
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            # Get image and label
            if hasattr(dataset, 'data'):
                # CIFAR format
                img = Image.fromarray(dataset.data[idx])
                label = dataset.targets[idx]
            else:
                img, label = dataset[idx]
                # If img is already a tensor, convert back to PIL
                if isinstance(img, torch.Tensor):
                    img = transforms.ToPILImage()(img)
            
            # Generate crops based on num_crops
            crops = []
            
            if num_crops == 1: # Single center crop
                img_resized = transforms.Resize(256)(img)
                crop = transforms.CenterCrop(crop_size)(img_resized)
                crop = transforms.ToTensor()(crop)
                crop = normalize(crop)
                crops.append(crop)
                
            elif num_crops == 10: # 5 crops (4 corners + center) with horizontal flip = 10 crops
                img_resized = transforms.Resize(256)(img)
                
                # Get 5 crops: 4 corners + center
                w, h = img_resized.size
                positions = [
                    (0, 0),                          # top-left
                    (w - crop_size, 0),              # top-right
                    (0, h - crop_size),              # bottom-left
                    (w - crop_size, h - crop_size),  # bottom-right
                    ((w - crop_size) // 2, (h - crop_size) // 2)  # center
                ]
                
                for x, y in positions:
                    crop = img_resized.crop((x, y, x + crop_size, y + crop_size))
                    # Original crop
                    crop_tensor = transforms.ToTensor()(crop)
                    crops.append(normalize(crop_tensor))
                    # Horizontally flipped crop
                    crop_flipped = transforms.functional.hflip(crop)
                    crop_flipped_tensor = transforms.ToTensor()(crop_flipped)
                    crops.append(normalize(crop_flipped_tensor))
                    
            elif num_crops == 144:
                # Exact 144-crop protocol from GoogLeNet paper:
                # 4 scales (shorter dimension = 256, 288, 320, 352)
                # × 3 positions (left/top, center, right/bottom squares)
                # × 6 crops per square (4 corners + center 224x224 + resized square to 224x224)
                # × 2 (original + horizontal flip)
                # = 4 × 3 × 6 × 2 = 144 crops
                
                scales = [256, 288, 320, 352]
                
                for scale in scales:
                    # Resize image so shorter dimension equals scale
                    w, h = img.size
                    if w < h:
                        # Width is shorter (portrait)
                        new_w = scale
                        new_h = int(h * scale / w)
                    else:
                        # Height is shorter (landscape)
                        new_h = scale
                        new_w = int(w * scale / h)
                    
                    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
                    w_resized, h_resized = img_resized.size
                    
                    # Extract 3 squares from the resized image
                    if w_resized < h_resized:
                        # Portrait: take top, center, bottom squares
                        square_size = w_resized
                        square_positions = [
                            0,                                    # top
                            (h_resized - square_size) // 2,      # center
                            h_resized - square_size              # bottom
                        ]
                        squares = [img_resized.crop((0, y, square_size, y + square_size)) 
                                  for y in square_positions]
                    else:
                        # Landscape: take left, center, right squares
                        square_size = h_resized
                        square_positions = [
                            0,                                    # left
                            (w_resized - square_size) // 2,      # center
                            w_resized - square_size              # right
                        ]
                        squares = [img_resized.crop((x, 0, x + square_size, square_size)) 
                                  for x in square_positions]
                    
                    # For each square, extract 6 crops
                    for square in squares:
                        square_crops = []
                        
                        # 1-4: Four corner crops (224x224)
                        corners = [
                            (0, 0),                                                    # top-left
                            (square_size - crop_size, 0),                             # top-right
                            (0, square_size - crop_size),                             # bottom-left
                            (square_size - crop_size, square_size - crop_size)        # bottom-right
                        ]
                        
                        for x, y in corners:
                            corner_crop = square.crop((x, y, x + crop_size, y + crop_size))
                            square_crops.append(corner_crop)
                        
                        # 5: Center crop (224x224)
                        center_x = (square_size - crop_size) // 2
                        center_y = (square_size - crop_size) // 2
                        center_crop = square.crop((center_x, center_y, 
                                                  center_x + crop_size, center_y + crop_size))
                        square_crops.append(center_crop)
                        
                        # 6: Entire square resized to 224x224
                        resized_crop = square.resize((crop_size, crop_size), Image.BILINEAR)
                        square_crops.append(resized_crop)
                        
                        # Add original and horizontally flipped versions
                        for crop in square_crops:
                            # Original
                            crop_tensor = transforms.ToTensor()(crop)
                            crops.append(normalize(crop_tensor))
                            # Horizontally flipped
                            crop_flipped = transforms.functional.hflip(crop)
                            crop_flipped_tensor = transforms.ToTensor()(crop_flipped)
                            crops.append(normalize(crop_flipped_tensor))
            
            # Stack crops and process in batches
            crops_tensor = torch.stack(crops).to(device)
            
            # Process crops in batches
            all_outputs = []
            for i in range(0, len(crops_tensor), batch_size):
                batch_crops = crops_tensor[i:i+batch_size]
                outputs = model(batch_crops)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                all_outputs.append(outputs)
            
            # Average predictions across all crops
            all_outputs = torch.cat(all_outputs, dim=0)
            avg_output = all_outputs.mean(dim=0, keepdim=True)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                if isinstance(label, torch.Tensor):
                    label_tensor = label.unsqueeze(0).to(device)
                else:
                    label_tensor = torch.tensor([label], dtype=torch.long).to(device)
                loss = criterion(avg_output, label_tensor)
                running_loss += loss.item()
            
            # Calculate top-1 and top-5 accuracy
            _, pred_top5 = avg_output.topk(5, dim=1, largest=True, sorted=True)
            pred_top1 = pred_top5[:, 0]
            
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            # Top-1 accuracy
            if pred_top1.item() == label:
                top1_correct += 1
            
            # Top-5 accuracy
            if label in pred_top5[0].cpu().numpy():
                top5_correct += 1
            
            total += 1
            
            # Print progress
            if (idx + 1) % 100 == 0:
                progress_str = (f"  Processed {idx + 1}/{len(dataset)} images - "
                               f"Top-1: {100.0 * top1_correct / total:.2f}%, "
                               f"Top-5: {100.0 * top5_correct / total:.2f}%")
                if criterion is not None:
                    progress_str += f", Loss: {running_loss / total:.4f}"
                print(progress_str)
    
    # Calculate accuracies and error rates
    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
    top1_err = 100.0 - top1_acc  # Top-1 error rate (as reported in paper)
    top5_err = 100.0 - top5_acc  # Top-5 error rate (as reported in paper)
    avg_loss = running_loss / total if criterion is not None else None
    
    # Print summary with both accuracy and error rates
    print(f"\nEvaluation Results ({num_crops}-crop):")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%  |  Top-1 Error: {top1_err:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%  |  Top-5 Error: {top5_err:.2f}%")
    if avg_loss is not None:
        print(f"  Average Loss: {avg_loss:.4f}")
    
    # Return as dictionary for easier access
    return {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'top1_err': top1_err,
        'top5_err': top5_err,
        'loss': avg_loss
    }



