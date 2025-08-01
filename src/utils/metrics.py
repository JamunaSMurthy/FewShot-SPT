"""
Metrics and evaluation functions for anomaly detection.
"""

import torch
import numpy as np
from sklearn.metrics import (
    auc, roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, accuracy_score, confusion_matrix,
    classification_report, f1_score
)
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from pathlib import Path


class AnomalyDetectionMetrics:
    """Compute various metrics for anomaly detection."""
    
    @staticmethod
    def compute_auc(predictions: np.ndarray, 
                   ground_truth: np.ndarray) -> float:
        """
        Compute AUC-ROC score.
        
        Args:
            predictions: (N,) - predicted anomaly scores [0, 1]
            ground_truth: (N,) - binary labels [0, 1]
        
        Returns:
            auc_score: float
        """
        return roc_auc_score(ground_truth, predictions)
    
    @staticmethod
    def compute_ap(predictions: np.ndarray,
                  ground_truth: np.ndarray) -> float:
        """Compute Average Precision."""
        return average_precision_score(ground_truth, predictions)
    
    @staticmethod
    def compute_accuracy(predictions: np.ndarray,
                        ground_truth: np.ndarray,
                        threshold: float = 0.5) -> float:
        """Compute classification accuracy."""
        pred_labels = (predictions > threshold).astype(int)
        return accuracy_score(ground_truth, pred_labels)
    
    @staticmethod
    def compute_f1(predictions: np.ndarray,
                  ground_truth: np.ndarray,
                  threshold: float = 0.5) -> float:
        """Compute F1 score."""
        pred_labels = (predictions > threshold).astype(int)
        return f1_score(ground_truth, pred_labels)
    
    @staticmethod
    def compute_confusion_matrix(predictions: np.ndarray,
                                ground_truth: np.ndarray,
                                threshold: float = 0.5) -> np.ndarray:
        """Compute confusion matrix."""
        pred_labels = (predictions > threshold).astype(int)
        return confusion_matrix(ground_truth, pred_labels)
    
    @staticmethod
    def compute_roc_curve(predictions: np.ndarray,
                         ground_truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
        """Compute ROC curve points."""
        fpr, tpr, thresholds = roc_curve(ground_truth, predictions)
        return fpr, tpr, thresholds
    
    @staticmethod
    def compute_pr_curve(predictions: np.ndarray,
                        ground_truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute precision-recall curve."""
        precision, recall, _ = precision_recall_curve(ground_truth, predictions)
        return precision, recall
    
    @staticmethod
    def compute_all_metrics(predictions: np.ndarray,
                           ground_truth: np.ndarray,
                           threshold: float = 0.5) -> Dict[str, float]:
        """Compute all metrics at once."""
        
        metrics = {
            'auc': AnomalyDetectionMetrics.compute_auc(predictions, ground_truth),
            'ap': AnomalyDetectionMetrics.compute_ap(predictions, ground_truth),
            'accuracy': AnomalyDetectionMetrics.compute_accuracy(
                predictions, ground_truth, threshold
            ),
            'f1': AnomalyDetectionMetrics.compute_f1(
                predictions, ground_truth, threshold
            ),
        }
        
        # Add per-class metrics
        pred_labels = (predictions > threshold).astype(int)
        tn, fp, fn, tp = AnomalyDetectionMetrics.compute_confusion_matrix(
            predictions, ground_truth, threshold
        ).ravel()
        
        metrics.update({
            'tp': float(tp),
            'fp': float(fp),
            'tn': float(tn),
            'fn': float(fn),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        })
        
        return metrics


class MeanAveragePrecision:
    """Compute Mean Average Precision for frame-level detection."""
    
    @staticmethod
    def compute_map(predictions: List[np.ndarray],
                   ground_truths: List[np.ndarray]) -> float:
        """
        Compute mAP across multiple videos.
        
        Args:
            predictions: List of prediction arrays per video
            ground_truths: List of ground truth arrays per video
        
        Returns:
            mean_ap: float
        """
        aps = []
        for pred, gt in zip(predictions, ground_truths):
            ap = average_precision_score(gt, pred)
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0


class TemporalMetrics:
    """Metrics specific to temporal/video anomaly detection."""
    
    @staticmethod
    def temporal_iou(pred_segments: List[Tuple],
                    gt_segments: List[Tuple]) -> float:
        """
        Compute temporal Intersection over Union.
        
        Args:
            pred_segments: List of (start, end) tuples for predictions
            gt_segments: List of (start, end) tuples for ground truth
        
        Returns:
            iou: float
        """
        def compute_iou(pred, gt):
            pred_start, pred_end = pred
            gt_start, gt_end = gt
            
            inter_start = max(pred_start, gt_start)
            inter_end = min(pred_end, gt_end)
            
            if inter_start >= inter_end:
                return 0
            
            intersection = inter_end - inter_start
            union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
            
            return intersection / union if union > 0 else 0
        
        # Match predictions to ground truth
        ious = [compute_iou(pred, gt) for pred, gt in zip(pred_segments, gt_segments)]
        return np.mean(ious) if ious else 0.0
    
    @staticmethod
    def frame_level_auc(frame_scores: np.ndarray,
                       frame_labels: np.ndarray) -> float:
        """Compute frame-level AUC."""
        return roc_auc_score(frame_labels, frame_scores)


class ComparisonMetrics:
    """Comparison metrics for few-shot learning evaluation."""
    
    @staticmethod
    def one_shot_accuracy(predictions: torch.Tensor,
                         labels: torch.Tensor) -> float:
        """Compute 1-shot accuracy."""
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return 100 * correct / total if total > 0 else 0
    
    @staticmethod
    def few_shot_accuracy(predictions: torch.Tensor,
                         labels: torch.Tensor,
                         k: int = 5) -> float:
        """Compute k-shot accuracy (top-k accuracy)."""
        _, top_k_pred = torch.topk(predictions, k, dim=1)
        labels_expanded = labels.unsqueeze(1).expand_as(top_k_pred)
        correct = (top_k_pred == labels_expanded).any(dim=1).sum().item()
        total = labels.size(0)
        return 100 * correct / total if total > 0 else 0


class MetricsVisualizer:
    """Visualize metrics and results."""
    
    @staticmethod
    def plot_roc_curve(fpr: np.ndarray,
                      tpr: np.ndarray,
                      auc_score: float,
                      save_path: str = None):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_pr_curve(precision: np.ndarray,
                     recall: np.ndarray,
                     ap_score: float,
                     save_path: str = None):
        """Plot Precision-Recall curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AP = {ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"PR curve saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray,
                             class_names: List[str],
                             save_path: str = None):
        """Plot confusion matrix."""
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()


def compute_video_level_metrics(frame_predictions: np.ndarray,
                               frame_labels: np.ndarray) -> Dict[str, float]:
    """
    Convert frame-level predictions to video-level scores.
    """
    
    # Aggregate frame-level scores to video-level
    video_scores = frame_predictions.mean()
    video_label = frame_labels.max()  # Video is positive if any frame is positive
    
    return {
        'video_score': video_scores,
        'video_label': video_label
    }


if __name__ == "__main__":
    print("Metrics module loaded successfully")
