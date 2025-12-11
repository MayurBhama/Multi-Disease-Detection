"""
src/training/ordinal_loss.py

Ordinal regression loss functions for Diabetic Retinopathy classification.
DR severity is ordinal: 0 < 1 < 2 < 3 < 4

Key losses:
- CORAL: Consistent Rank Logits loss
- Ordinal Cross-Entropy: Extended cross-entropy for ordinal targets
- Focal Ordinal Loss: Combines focal loss with ordinal regression
"""

import tensorflow as tf
import numpy as np


def ordinal_softmax_loss(y_true, y_pred, num_classes=5, weight_type='linear'):
    """
    Ordinal-aware softmax cross-entropy loss.
    Penalizes predictions based on distance from true class.
    
    Args:
        y_true: One-hot encoded true labels (batch, num_classes)
        y_pred: Softmax probabilities (batch, num_classes)
        num_classes: Number of classes
        weight_type: 'linear', 'quadratic', or 'none'
    """
    # Cast to float32 for mixed precision compatibility
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # Get true class index
    true_class = tf.argmax(y_true, axis=-1)  # (batch,)
    
    # Create ordinal distance weights
    class_indices = tf.range(num_classes, dtype=tf.float32)
    
    if weight_type == 'linear':
        # Weight increases linearly with distance
        distances = tf.abs(tf.cast(true_class[:, None], tf.float32) - class_indices)
        weights = 1.0 + distances
    elif weight_type == 'quadratic':
        # QWK-inspired quadratic weighting
        distances = tf.abs(tf.cast(true_class[:, None], tf.float32) - class_indices)
        weights = 1.0 + tf.square(distances)
    else:
        weights = tf.ones((tf.shape(y_true)[0], num_classes), dtype=tf.float32)
    
    # Weighted cross-entropy
    weighted_ce = -weights * y_true * tf.math.log(y_pred)
    
    return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))


def coral_loss(y_true, logits, num_classes=5):
    """
    CORAL (Consistent Rank Logits) loss for ordinal regression.
    From "Rank consistent ordinal regression for neural networks" (Cao et al. 2020)
    
    The model predicts K-1 binary classifiers: P(y > k) for k=0,...,K-2
    
    Args:
        y_true: One-hot encoded labels (batch, num_classes)
        logits: Raw logits for K-1 binary tasks (batch, num_classes-1)
    """
    # Convert one-hot to class index
    true_class = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)  # (batch,)
    
    num_thresholds = num_classes - 1
    
    # Create binary targets: 1 if y > k, 0 otherwise
    # For each sample, create targets for all K-1 thresholds
    thresholds = tf.range(num_thresholds, dtype=tf.float32)  # [0, 1, 2, 3]
    
    # Binary targets: (batch, num_thresholds)
    # y_binary[i, k] = 1 if true_class[i] > k
    y_binary = tf.cast(true_class[:, None] > thresholds, tf.float32)
    
    # Binary cross-entropy for each threshold
    # logits are raw scores, apply sigmoid
    probs = tf.sigmoid(logits)
    probs = tf.clip_by_value(probs, 1e-7, 1 - 1e-7)
    
    bce = -y_binary * tf.math.log(probs) - (1 - y_binary) * tf.math.log(1 - probs)
    
    return tf.reduce_mean(tf.reduce_sum(bce, axis=-1))


def ordinal_focal_loss(y_true, y_pred, gamma=2.0, alpha=None, num_classes=5):
    """
    Focal loss adapted for ordinal classification.
    Combines focal loss with ordinal distance weighting.
    
    Args:
        y_true: One-hot encoded labels (batch, num_classes)
        y_pred: Softmax probabilities (batch, num_classes)
        gamma: Focal loss focusing parameter
        alpha: Per-class weights (list or dict), or None for uniform
    """
    # Cast to float32 for mixed precision compatibility
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    
    # Class weights
    if alpha is None:
        alpha_tensor = tf.ones(num_classes, dtype=tf.float32)
    elif isinstance(alpha, dict):
        alpha_tensor = tf.constant([alpha[i] for i in range(num_classes)], dtype=tf.float32)
    else:
        alpha_tensor = tf.constant(alpha, dtype=tf.float32)
    
    # Get true class for ordinal weighting
    true_class = tf.argmax(y_true, axis=-1)
    class_indices = tf.range(num_classes, dtype=tf.float32)
    
    # Quadratic distance penalty (like QWK)
    distances = tf.square(tf.cast(true_class[:, None], tf.float32) - class_indices)
    ordinal_weights = 1.0 + 0.5 * distances / tf.cast(num_classes - 1, tf.float32) ** 2
    
    # Focal modulation
    pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
    focal_weight = tf.pow(1 - pt, gamma)
    
    # Combined loss
    ce = -y_true * tf.math.log(y_pred)
    weighted_ce = alpha_tensor * ordinal_weights * focal_weight * ce
    
    return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))


class OrdinalFocalLoss(tf.keras.losses.Loss):
    """Keras Loss class wrapper for ordinal focal loss."""
    
    def __init__(self, gamma=2.0, alpha=None, num_classes=5, name='ordinal_focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
    
    def call(self, y_true, y_pred):
        return ordinal_focal_loss(y_true, y_pred, self.gamma, self.alpha, self.num_classes)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'num_classes': self.num_classes
        })
        return config


class CombinedOrdinalLoss(tf.keras.losses.Loss):
    """
    Combined loss for ordinal classification:
    L = lambda_focal * FocalLoss + lambda_ordinal * OrdinalDistanceLoss
    """
    
    def __init__(
        self,
        focal_gamma=2.0,
        class_weights=None,
        ordinal_weight=0.5,
        label_smoothing=0.1,
        num_classes=5,
        name='combined_ordinal_loss'
    ):
        super().__init__(name=name)
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights
        self.ordinal_weight = ordinal_weight
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true_smooth = y_true * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        else:
            y_true_smooth = y_true
        
        # Focal loss component
        alpha = self.class_weights if self.class_weights else [1.0] * self.num_classes
        focal_loss = ordinal_focal_loss(y_true_smooth, y_pred, self.focal_gamma, alpha, self.num_classes)
        
        # Ordinal distance component
        ordinal_loss = ordinal_softmax_loss(y_true, y_pred, self.num_classes, 'quadratic')
        
        return focal_loss + self.ordinal_weight * ordinal_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'focal_gamma': self.focal_gamma,
            'class_weights': self.class_weights,
            'ordinal_weight': self.ordinal_weight,
            'label_smoothing': self.label_smoothing,
            'num_classes': self.num_classes
        })
        return config


def qwk_metric(y_true, y_pred, num_classes=5):
    """
    Compute Quadratic Weighted Kappa in TensorFlow.
    This is a differentiable approximation for use as a metric.
    """
    y_true_idx = tf.argmax(y_true, axis=-1)
    y_pred_idx = tf.argmax(y_pred, axis=-1)
    
    # Compute confusion matrix weights
    w = tf.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            w = tf.tensor_scatter_nd_update(
                w, [[i, j]], 
                [tf.square(tf.cast(i - j, tf.float32)) / tf.square(tf.cast(num_classes - 1, tf.float32))]
            )
    
    # Compute histograms
    O = tf.math.confusion_matrix(y_true_idx, y_pred_idx, num_classes=num_classes)
    O = tf.cast(O, tf.float32)
    
    # Normalize
    N = tf.reduce_sum(O)
    O = O / N
    
    # Expected matrix
    act_hist = tf.reduce_sum(O, axis=1)
    pred_hist = tf.reduce_sum(O, axis=0)
    E = tf.tensordot(act_hist, pred_hist, axes=0)
    
    # QWK
    numerator = tf.reduce_sum(w * O)
    denominator = tf.reduce_sum(w * E)
    
    qwk = 1 - numerator / (denominator + 1e-8)
    
    return qwk


class QWKMetric(tf.keras.metrics.Metric):
    """Keras Metric class for Quadratic Weighted Kappa."""
    
    def __init__(self, num_classes=5, name='qwk', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name='cm',
            shape=(num_classes, num_classes),
            initializer='zeros'
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_idx = tf.argmax(y_true, axis=-1)
        y_pred_idx = tf.argmax(y_pred, axis=-1)
        
        cm = tf.math.confusion_matrix(
            y_true_idx, y_pred_idx, 
            num_classes=self.num_classes,
            dtype=tf.float32
        )
        
        self.confusion_matrix.assign_add(cm)
    
    def result(self):
        O = self.confusion_matrix
        N = tf.reduce_sum(O)
        O = O / (N + 1e-8)
        
        # Weight matrix
        w = tf.zeros((self.num_classes, self.num_classes))
        indices = []
        values = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                indices.append([i, j])
                values.append(float((i - j) ** 2) / float((self.num_classes - 1) ** 2))
        w = tf.scatter_nd(indices, values, (self.num_classes, self.num_classes))
        
        # Expected
        act_hist = tf.reduce_sum(O, axis=1)
        pred_hist = tf.reduce_sum(O, axis=0)
        E = tf.tensordot(act_hist, pred_hist, axes=0)
        
        numerator = tf.reduce_sum(w * O)
        denominator = tf.reduce_sum(w * E)
        
        return 1 - numerator / (denominator + 1e-8)
    
    def reset_state(self):
        self.confusion_matrix.assign(tf.zeros((self.num_classes, self.num_classes)))
