"""
Adaptive Prototypical Few-Shot Learning
Improves generalization to unseen anomalies using prototypical networks and contrastive learning.
Adapts prototypes based on task-specific information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import math


class PrototypeGenerator(nn.Module):
    """Generates class prototypes from support samples."""
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        """
        Generate prototype as mean of support features.
        
        Args:
            support_features: (K, C) - support set features for single class
        Returns:
            prototype: (C,) - class prototype
        """
        # Compute mean of support features
        mean_prototype = support_features.mean(dim=0)
        
        # Optionally refine through network
        refined = self.feature_processor(mean_prototype.unsqueeze(0))
        
        return refined.squeeze(0)


class ContrastivePrototypeLearner(nn.Module):
    """Learns prototypes with contrastive objectives."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 temperature: float = 0.05):
        super().__init__()
        
        self.temperature = temperature
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Prototype generator
        self.prototype_generator = PrototypeGenerator(feature_dim, hidden_dim)
        
    def forward(self, support_features: torch.Tensor,
                query_features: torch.Tensor,
                support_labels: torch.Tensor,
                query_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Learn prototypes with contrastive loss.
        
        Args:
            support_features: (N_support, C) - support set features
            query_features: (N_query, C) - query set features
            support_labels: (N_support,) - support set labels
            query_labels: (N_query,) - query set labels (optional)
        
        Returns:
            prototypes: (num_classes, C) - class prototypes
            losses: dict of loss values
        """
        # Project features to contrastive space
        support_proj = self.projection_head(support_features)  # (N_s, C)
        query_proj = self.projection_head(query_features)      # (N_q, C)
        
        # Generate prototypes for each class
        unique_labels = torch.unique(support_labels)
        num_classes = len(unique_labels)
        
        prototypes = []
        for label in unique_labels:
            class_mask = (support_labels == label)
            class_features = support_features[class_mask]
            prototype = self.prototype_generator(class_features)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (num_classes, C)
        
        # Compute contrastive loss
        losses = self._compute_contrastive_losses(
            support_proj, query_proj, 
            support_labels, query_labels,
            prototypes
        )
        
        return prototypes, losses
    
    def _compute_contrastive_losses(self, 
                                   support_proj: torch.Tensor,
                                   query_proj: torch.Tensor,
                                   support_labels: torch.Tensor,
                                   query_labels: Optional[torch.Tensor],
                                   prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute various contrastive loss terms."""
        losses = {}
        
        # Supervised contrastive loss using support set
        loss_support = self._supervised_contrastive_loss(
            support_proj, support_labels, temperature=self.temperature
        )
        losses['support_contrastive'] = loss_support
        
        # Instance discrimination loss
        if query_labels is not None:
            loss_query = self._supervised_contrastive_loss(
                query_proj, query_labels, temperature=self.temperature
            )
            losses['query_contrastive'] = loss_query
        
        return losses
    
    @staticmethod
    def _supervised_contrastive_loss(features: torch.Tensor,
                                    labels: torch.Tensor,
                                    temperature: float = 0.05) -> torch.Tensor:
        """
        Supervised contrastive loss.
        Reference: https://arxiv.org/abs/2004.11362
        """
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Compute pairwise similarities
        logits = torch.matmul(features, features.T) / temperature
        
        # Create label mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity from negative examples
        diag_mask = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
        mask_pos_neg = mask.clone()
        mask_pos_neg[diag_mask] = 0
        
        # Apply softmax
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Compute loss
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        loss = -mean_log_prob_pos.mean()
        
        return loss


class AdaptivePrototypeRefiner(nn.Module):
    """Refines prototypes adaptively based on query samples."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 num_refinement_steps: int = 3):
        super().__init__()
        
        self.num_refinement_steps = num_refinement_steps
        
        # Refinement module
        self.refinement_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim)
        )
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, prototypes: torch.Tensor,
                query_features: torch.Tensor,
                query_distances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine prototypes based on query samples.
        
        Args:
            prototypes: (num_classes, C) - initial prototypes
            query_features: (N_q, C) - query features
            query_distances: (N_q, num_classes) - distances to prototypes
        
        Returns:
            refined_prototypes: (num_classes, C) - refined prototypes
            confidence_scores: (num_classes, 1) - confidence in prototypes
        """
        num_classes = prototypes.shape[0]
        refined_prototypes = prototypes.clone()
        confidence_scores = []
        
        for step in range(self.num_refinement_steps):
            # Compute soft assignments
            soft_assignments = F.softmax(-query_distances / (step + 1), dim=-1)  # (N_q, num_classes)
            
            # Update each prototype
            for c in range(num_classes):
                assignment = soft_assignments[:, c:c+1]  # (N_q, 1)
                weighted_features = (query_features * assignment).sum(dim=0, keepdim=True)
                weight_sum = assignment.sum(dim=0, keepdim=True).clamp(min=1e-6)
                weighted_mean = weighted_features / weight_sum
                
                # Compute refinement
                concat_features = torch.cat([refined_prototypes[c:c+1], weighted_mean], dim=-1)
                refinement = self.refinement_net(concat_features)
                
                # Update prototype with refinement
                refined_prototypes[c] = refined_prototypes[c] + 0.1 * refinement.squeeze(0)
        
        # Compute confidence scores
        for c in range(num_classes):
            concat = torch.cat([prototypes[c:c+1].expand_as(query_features), query_features], dim=-1)
            confidence = self.confidence_net(concat).mean()
            confidence_scores.append(confidence)
        
        confidence_scores = torch.stack(confidence_scores)
        
        return refined_prototypes, confidence_scores


class AdaptivePrototypicalFewShotLearning(nn.Module):
    """
    Adaptive Prototypical Few-Shot Learning (APFSL) module.
    Combines prototypical networks with adaptive refinement and contrastive learning.
    
    Paper contribution: Improves generalization to unseen anomalies with adaptivity.
    """
    
    def __init__(self,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 num_refinement_steps: int = 3,
                 temperature: float = 0.05,
                 decay: float = 0.99):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.decay = decay
        
        # Core components
        self.prototype_generator = PrototypeGenerator(feature_dim, hidden_dim)
        self.contrastive_learner = ContrastivePrototypeLearner(
            feature_dim, hidden_dim, temperature
        )
        self.prototype_refiner = AdaptivePrototypeRefiner(
            feature_dim, num_refinement_steps
        )
        
        # Feature normalization
        self.feature_normalizer = nn.LayerNorm(feature_dim)
        
        # Learnable temperature for distance computation
        self.distance_temperature = nn.Parameter(torch.tensor(1.0))
        
        # Support memory for episodic training
        self.register_buffer('memory_prototypes', None)
        self.register_buffer('memory_counts', None)
    
    def forward(self, 
                support_features: torch.Tensor,
                support_labels: torch.Tensor,
                query_features: torch.Tensor,
                query_labels: Optional[torch.Tensor] = None,
                return_distances: bool = False,
                adaptive_refinement: bool = True) -> torch.Tensor or Tuple:
        """
        Few-shot learning with adaptive prototypes.
        
        Args:
            support_features: (N_support, C) - support set features
            support_labels: (N_support,) - support set labels
            query_features: (N_query, C) - query set features
            query_labels: (N_query,) - query set labels (optional)
            return_distances: if True, return query-to-prototype distances
            adaptive_refinement: if True, apply adaptive refinement
        
        Returns:
            logits: (N_query, num_classes) - classification logits
            or (logits, distances, refined_prototypes) if return_distances/refinement flags
        """
        # Normalize features
        support_normalized = self.feature_normalizer(support_features)
        query_normalized = self.feature_normalizer(query_features)
        
        # Generate prototypes using contrastive learning
        prototypes, losses = self.contrastive_learner(
            support_normalized, query_normalized,
            support_labels, query_labels
        )
        prototypes = self.feature_normalizer(prototypes)
        
        # Compute distances from query to prototypes
        distances = self._compute_distances(
            query_normalized, prototypes
        )  # (N_query, num_classes)
        
        # Adaptive prototype refinement
        if adaptive_refinement:
            prototypes, confidence = self.prototype_refiner(
                prototypes, query_normalized, distances
            )
            prototypes = self.feature_normalizer(prototypes)
            
            # Recompute distances with refined prototypes
            distances = self._compute_distances(
                query_normalized, prototypes
            )
        
        # Convert distances to logits
        logits = -distances * self.distance_temperature  # (N_query, num_classes)
        
        # Update episodic memory if available
        if self._should_update_memory():
            self._update_memory(prototypes, support_labels)
        
        # Return based on flags
        if return_distances:
            return logits, distances, prototypes, losses
        
        return logits
    
    def _compute_distances(self, 
                          query_features: torch.Tensor,
                          prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute distances from queries to prototypes.
        Uses Euclidean distance with learnable temperature scaling.
        """
        # (N_query, 1, C) - (1, num_classes, C) = (N_query, num_classes, C)
        diffs = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
        distances = torch.norm(diffs, dim=-1)  # (N_query, num_classes)
        return distances
    
    def _should_update_memory(self) -> bool:
        """Check if memory should be updated."""
        return self.memory_prototypes is not None
    
    def _update_memory(self, prototypes: torch.Tensor, labels: torch.Tensor):
        """Update episodic memory with exponential moving average."""
        if self.memory_prototypes is None:
            self.memory_prototypes = prototypes.clone().detach()
            self.memory_counts = torch.ones(prototypes.shape[0], device=prototypes.device)
        else:
            # EMA update
            self.memory_prototypes = (
                self.decay * self.memory_prototypes +
                (1 - self.decay) * prototypes.detach()
            )
            self.memory_counts += 1
    
    def get_prototypes(self) -> torch.Tensor:
        """Get current prototypes (from memory if available)."""
        if self.memory_prototypes is not None:
            return self.memory_prototypes
        return None
    
    def _compute_contrastive_loss(self,
                                 features: torch.Tensor,
                                 labels: torch.Tensor) -> torch.Tensor:
        """Contrastive loss for feature learning."""
        # Normalize
        features = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T)
        
        # Create positive pairs mask
        expanded_labels = labels.unsqueeze(1).expand(-1, len(labels))
        mask = (expanded_labels == expanded_labels.T).float()
        
        # Remove self-similarity
        mask.fill_diagonal_(0)
        
        # Compute loss
        pos_mask = mask > 0
        neg_mask = (mask == 0) & (~torch.eye(len(features), dtype=torch.bool, device=features.device))
        
        pos_sim = similarity[pos_mask].reshape(features.shape[0], -1)
        neg_sim = similarity[neg_mask].reshape(features.shape[0], -1)
        
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels_loss = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        
        loss = F.cross_entropy(logits / self.temperature, labels_loss)
        return loss


class FewShotClassifier(nn.Module):
    """Wrapper for few-shot classification with various strategies."""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 learning_rate: float = 0.001,
                 use_adaptive: bool = True):
        super().__init__()
        
        self.learner = AdaptivePrototypicalFewShotLearning(
            feature_dim=feature_dim,
            hidden_dim=256,
            num_refinement_steps=3,
            temperature=0.05
        )
        self.use_adaptive = use_adaptive
        
    def forward(self, support_set, query_set):
        """
        Args:
            support_set: dict with 'features' (N_s, C) and 'labels' (N_s,)
            query_set: dict with 'features' (N_q, C) and 'labels' (N_q,) [optional]
        Returns:
            predictions: (N_q, num_classes)
        """
        support_features = support_set['features']
        support_labels = support_set['labels']
        query_features = query_set['features']
        query_labels = query_set.get('labels', None)
        
        logits = self.learner(
            support_features, support_labels,
            query_features, query_labels,
            adaptive_refinement=self.use_adaptive
        )
        
        return torch.softmax(logits, dim=-1)


# Factory function
def create_apfsl(feature_dim: int = 512) -> AdaptivePrototypicalFewShotLearning:
    """Factory function for APFSL module."""
    return AdaptivePrototypicalFewShotLearning(
        feature_dim=feature_dim,
        hidden_dim=256,
        num_refinement_steps=3,
        temperature=0.05,
        decay=0.99
    )
