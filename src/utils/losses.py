"""
Loss functions for training FewShot-SPT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Contrastive learning loss for few-shot learning.
    Based on SimCLR and supervised contrastive learning papers.
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features: torch.Tensor,
               labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            features: (B, D) - feature vectors
            labels: (B,) - class labels (optional for supervised contrastive)
        Returns:
            loss: scalar
        """
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)  # (B, B)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create positive pair mask
        if labels is not None:
            # Supervised contrastive loss
            labels = labels.unsqueeze(1)
            mask = (labels == labels.T).float()
            
            # Remove self-similarity from negatives
            mask_no_diag = mask.clone()
            mask_no_diag.fill_diagonal_(0)
        else:
            # Unsupervised: assume consecutive pairs
            B = features.shape[0]
            mask = torch.zeros(B, B, device=features.device)
            # Example: 0-1, 2-3, 4-5 are pairs, 1-0, 3-2, 5-4
            for i in range(0, B, 2):
                if i + 1 < B:
                    mask[i, i+1] = 1
                    mask[i+1, i] = 1
            mask_no_diag = mask.clone()
            mask_no_diag.fill_diagonal_(0)
        
        # Compute log_prob
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Compute loss
        mean_log_prob_pos = (mask_no_diag * log_prob).sum(dim=1) / (mask_no_diag.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class PrototypicalLoss(nn.Module):
    """
    Prototypical Networks loss for few-shot learning.
    """
    
    def __init__(self, temperature: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, query_features: torch.Tensor,
               query_labels: torch.Tensor,
               prototype_features: torch.Tensor,
               prototype_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_features: (N_q, D) - query sample features
            query_labels: (N_q,) - query labels
            prototype_features: (N_p, D) - prototype features
            prototype_labels: (N_p,) - prototype class labels
        Returns:
            loss: scalar
        """
        
        # Normalize features
        query_norm = F.normalize(query_features, dim=1)
        prototype_norm = F.normalize(prototype_features, dim=1)
        
        # Compute distances
        # (N_q, 1, D) - (1, N_p, D) = (N_q, N_p, D)
        dists = torch.norm(
            query_norm.unsqueeze(1) - prototype_norm.unsqueeze(0),
            dim=2
        )
        
        # Convert distances to log probabilities (lower distance = higher prob)
        log_probs = F.log_softmax(-dists * self.temperature, dim=1)
        
        # Target: which prototype for each query
        # Create target indices based on labels
        unique_labels = torch.unique(prototype_labels)
        target_indices = []
        for q_label in query_labels:
            for idx, p_label in enumerate(prototype_labels):
                if q_label == p_label:
                    target_indices.append(idx)
                    break
        
        target = torch.tensor(target_indices, device=query_features.device)
        
        # Compute NLL loss
        loss = F.nll_loss(log_probs, target, reduction=self.reduction)
        
        return loss


class AnomalyLoss(nn.Module):
    """
    Combines multiple losses for anomaly detection.
    """
    
    def __init__(self, 
                temperature: float = 0.07,
                lambda_contrastive: float = 1.0,
                lambda_classification: float = 1.0,
                lambda_prototype: float = 0.5):
        super().__init__()
        
        self.lambda_contrastive = lambda_contrastive
        self.lambda_classification = lambda_classification
        self.lambda_prototype = lambda_prototype
        
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.classification_loss = nn.CrossEntropyLoss()
        self.prototype_loss = PrototypicalLoss(temperature=temperature)
    
    def forward(self,
               logits: Optional[torch.Tensor] = None,
               labels: Optional[torch.Tensor] = None,
               features: Optional[torch.Tensor] = None,
               query_features: Optional[torch.Tensor] = None,
               query_labels: Optional[torch.Tensor] = None,
               prototypes: Optional[torch.Tensor] = None,
               prototype_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Returns:
            dict with loss components
        """
        
        losses = {}
        total_loss = 0
        
        # Classification loss
        if logits is not None and labels is not None:
            loss_clf = self.classification_loss(logits, labels)
            losses['classification'] = loss_clf
            total_loss += self.lambda_classification * loss_clf
        
        # Contrastive loss
        if features is not None and labels is not None:
            loss_cont = self.contrastive_loss(features, labels)
            losses['contrastive'] = loss_cont
            total_loss += self.lambda_contrastive * loss_cont
        
        # Prototypical loss
        if (query_features is not None and query_labels is not None and
            prototypes is not None and prototype_labels is not None):
            loss_proto = self.prototype_loss(
                query_features, query_labels,
                prototypes, prototype_labels
            )
            losses['prototypical'] = loss_proto
            total_loss += self.lambda_prototype * loss_proto
        
        losses['total'] = total_loss
        
        return losses


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) - model outputs
            targets: (B,) - target labels
        Returns:
            loss: scalar
        """
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute focal term
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma
        
        # Apply focal weighting
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedAnomalyLoss(nn.Module):
    """
    Advanced loss combining focal loss, contrastive learning, and anomaly-specific terms.
    """
    
    def __init__(self,
                focal_alpha: float = 0.25,
                focal_gamma: float = 2.0,
                contrastive_temperature: float = 0.07,
                lambda_focal: float = 1.0,
                lambda_contrastive: float = 0.5,
                lambda_anomaly: float = 0.3):
        super().__init__()
        
        self.lambda_focal = lambda_focal
        self.lambda_contrastive = lambda_contrastive
        self.lambda_anomaly = lambda_anomaly
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self,
               logits: torch.Tensor,
               targets: torch.Tensor,
               features: Optional[torch.Tensor] = None,
               anomaly_scores: Optional[torch.Tensor] = None) -> Dict:
        """
        Compute combined loss.
        
        Args:
            logits: (B, C) - classification logits
            targets: (B,) - target labels
            features: (B, D) - feature representations
            anomaly_scores: (B,) - per-sample anomaly scores
        
        Returns:
            dict with loss components
        """
        
        losses = {}
        
        # Focal loss (handles class imbalance)
        loss_focal = self.focal_loss(logits, targets)
        losses['focal'] = loss_focal
        
        # Contrastive loss if features provided
        if features is not None:
            loss_contrastive = self.contrastive_loss(features, targets)
            losses['contrastive'] = loss_contrastive
        else:
            loss_contrastive = 0
        
        # Anomaly score regularization
        if anomaly_scores is not None:
            # Penalize predictions misaligned with anomaly scores
            anomaly_targets = targets.float()
            loss_anomaly = F.binary_cross_entropy(
                torch.sigmoid(anomaly_scores),
                anomaly_targets
            )
            losses['anomaly_reg'] = loss_anomaly
        else:
            loss_anomaly = 0
        
        # Total loss
        total_loss = (
            self.lambda_focal * loss_focal +
            self.lambda_contrastive * loss_contrastive +
            self.lambda_anomaly * loss_anomaly
        )
        
        losses['total'] = total_loss
        
        return losses


if __name__ == "__main__":
    print("Loss functions module loaded successfully")
