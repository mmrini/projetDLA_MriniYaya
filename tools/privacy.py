"""
Privacy threat analysis tools for GoogLeNet. Includes Membership Inference Attack.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


class MembershipInferenceAttack:
    """
    Membership Inference Attack to determine if a sample was in the training set.
    
    Uses a threshold-based approach on prediction confidence to distinguish between
    training members and non-members.
    """
    
    def __init__(self):
        """
        Initialize the threshold-based membership inference attack.
        """
        self.threshold = None

        
    def prepare_attack_data(self, model, member_loader, non_member_loader, device):
        
        """ 
        Prepare data for training the attack model.
        
        Args:
            model: Target model to attack
            member_loader: DataLoader with training (member) samples
            non_member_loader: DataLoader with non-training (non-member) samples
            device: Device to use
        
        Returns:
            X_attack: Maximum confidence scores for each sample (N,)
            y_attack: Membership labels (N,) - 1 for member, 0 for non-member
         """
        model.eval()
        
        features_member = []
        labels_member = []
        features_non_member = []
        labels_non_member = []
        
        # Extract features from members
        with torch.no_grad():
            for images, labels in member_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Use softmax probabilities as features
                probs = F.softmax(outputs, dim=1)
                features_member.append(probs.cpu().numpy())
                labels_member.append(labels.cpu().numpy())
        
        # Extract features from non-members
        with torch.no_grad():
            for images, labels in non_member_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = F.softmax(outputs, dim=1)
                features_non_member.append(probs.cpu().numpy())
                labels_non_member.append(labels.cpu().numpy())
        
        # Concatenate
        features_member = np.vstack(features_member)
        labels_member = np.concatenate(labels_member)
        features_non_member = np.vstack(features_non_member)
        labels_non_member = np.concatenate(labels_non_member)
        
        # Use max confidence as the attack feature
        max_conf_member = features_member.max(axis=1)
        max_conf_non = features_non_member.max(axis=1)
        
        # Create attack dataset: confidence scores and membership labels
        X_attack = np.concatenate([max_conf_member, max_conf_non])
        y_attack = np.concatenate([
            np.ones(len(max_conf_member)),
            np.zeros(len(max_conf_non))
        ])
        
        return X_attack, y_attack 
    
    def train_attack(self, X_attack, y_attack):
        """
        Train the threshold-based attack by finding the optimal confidence threshold.
        
        Args:
            X_attack: Confidence scores array (N,)
            y_attack: Membership labels (N,) - 1 for member, 0 for non-member
        
        Returns:
            threshold: Optimal threshold for membership decision
        """
        # Find optimal threshold using ROC curve
        fpr, tpr, thresholds = roc_curve(y_attack, X_attack) #X_attack)
        
        # Youden's index to find optimal threshold (maximizes TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        self.threshold = optimal_threshold
        print(f"Optimal threshold found: {optimal_threshold:.4f}")
        return optimal_threshold
    
    def predict(self, X_test):
        """
        Predict membership for test samples.
        
        Args:
            X_test: Confidence scores (N,)
        
        Returns:
            predictions: Membership predictions (N,) - 1 for member, 0 for non-member
            scores: Confidence scores (N,)
        """
        if self.threshold is None:
            raise ValueError("Attack model not trained. Call train_attack() first.")
        
        scores = X_test
        predictions = (scores >= self.threshold).astype(int)
        
        return predictions , scores
    
    def evaluate(self, model, member_loader, non_member_loader, device):
        """
        Evaluate the membership inference attack.
        
        Args:
            model: Target model
            member_loader: DataLoader with member samples
            non_member_loader: DataLoader with non-member samples
            device: Device to use
        
        Returns:
            dict with evaluation metrics
        """
        # Prepare test data
        X_test, y_test = self.prepare_attack_data(model, member_loader, non_member_loader, device)
        
        # Predict
        predictions, scores = self.predict(X_test)
        
        # Calculate metrics
        accuracy = (predictions == y_test).mean()
        auc = roc_auc_score(y_test, scores)
        
        # Precision and recall
        precision, recall, _ = precision_recall_curve(y_test, scores)
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'predictions': predictions,
            'scores': scores,
            'true_labels': y_test
        }
    

def plot_membership_inference_results(results, save_path=None):
    """
    Plot ROC curve and confidence distributions for membership inference.
    
    Args:
        results: Dict from MembershipInferenceAttack.evaluate()
        save_path: Path to save the plot (optional)
    """
    from scipy.stats import gaussian_kde
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(results['true_labels'], results['scores'])
    axes[0].plot(fpr, tpr, label=f"AUC = {results['auc']:.3f}", linewidth=2, color='blue')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve - Membership Inference Attack', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # Confidence distributions with histograms AND density curves
    member_scores = results['scores'][results['true_labels'] == 1]
    non_member_scores = results['scores'][results['true_labels'] == 0]
    
    # Histograms (transparents)
    axes[1].hist(member_scores, bins=50, alpha=0.4, label='Members (histogram)', 
                 density=True, color='blue', edgecolor='blue', linewidth=0.5)
    axes[1].hist(non_member_scores, bins=50, alpha=0.4, label='Non-members (histogram)', 
                 density=True, color='red', edgecolor='red', linewidth=0.5)
    
    # Density curves (KDE) - smooth lines on top
    xs = np.linspace(min(results['scores'].min(), 0), 1, 200)
    
    if len(member_scores) > 1:
        density_members = gaussian_kde(member_scores)
        axes[1].plot(xs, density_members(xs), label='Members (density)', 
                     linewidth=2.5, color='blue')
    
    if len(non_member_scores) > 1:
        density_non_members = gaussian_kde(non_member_scores)
        axes[1].plot(xs, density_non_members(xs), label='Non-members (density)', 
                     linewidth=2.5, color='red')
    
    axes[1].set_xlabel('Prediction Confidence / Attack Score', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


