# train_efficientnet_ensemble_v2.py
"""
Industry-Grade Diabetic Retinopathy Classifier
==============================================
EfficientNet Ensemble:
- Model 1: EfficientNetV2-S
- Model 2: EfficientNetB2
- Model 3: EfficientNetB0

Features:
- Deep parameter monitoring (QWK, AUC, F1, Loss, Accuracy per epoch)
- TensorBoard integration
- Model checkpointing with best weights
- Per-class metrics tracking
- Full logging and exception handling

Target: QWK >= 0.90, AUC >= 0.95
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_auc_score, f1_score, accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
from src.utils.exception import CustomException, DataLoadError

logger = get_logger(__name__)


# =====================================================
# CONFIGURATION
# =====================================================
class Config:
    """Central configuration for training."""
    TRAIN_CSV = "data/raw/retina/train.csv"
    IMAGE_DIR = "data/raw/retina/train_images"
    OUTPUT_DIR = "outputs/production"
    MODEL_DIR = "outputs/production/models"
    GRAPH_DIR = "outputs/production/graphs"
    LOG_DIR = "logs/tensorboard"
    
    IMAGE_SIZE = (224, 224)  # Standard for EfficientNet
    BATCH_SIZE = 16
    NUM_CLASSES = 5
    SEED = 42
    
    HEAD_EPOCHS = 10
    FINETUNE_EPOCHS = 20
    HEAD_LR = 1e-3
    FINETUNE_LR = 1e-4
    
    CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    TARGET_QWK = 0.90
    TARGET_AUC = 0.95


# Create directories
for d in [Config.OUTPUT_DIR, Config.MODEL_DIR, Config.GRAPH_DIR, Config.LOG_DIR]:
    os.makedirs(d, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE


# =====================================================
# PREPROCESSING - FIXED FOR EFFICIENTNET
# =====================================================
def preprocess_image_v2s(image_path, target_size=Config.IMAGE_SIZE):
    """Preprocessing for EfficientNetV2-S."""
    try:
        if isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8')
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop black borders
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            margin = 5
            x, y = max(0, x - margin), max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            img = img[y:y+h, x:x+w]
        
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # CORRECT preprocessing for EfficientNetV2
        img = img.astype(np.float32)
        img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
        
        return img
    except Exception as e:
        logger.error(f"Preprocessing failed for {image_path}: {e}")
        raise


def preprocess_image_b_series(image_path, target_size=Config.IMAGE_SIZE):
    """Preprocessing for EfficientNetB0/B2."""
    try:
        if isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8')
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop black borders
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            margin = 5
            x, y = max(0, x - margin), max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            img = img[y:y+h, x:x+w]
        
        # Resize
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # CORRECT preprocessing for EfficientNetB-series
        img = img.astype(np.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        
        return img
    except Exception as e:
        logger.error(f"Preprocessing failed for {image_path}: {e}")
        raise


# =====================================================
# DATA LOADING
# =====================================================
def load_data():
    """Load data with aggressive oversampling."""
    try:
        logger.info("=" * 60)
        logger.info("LOADING DATA")
        logger.info("=" * 60)
        
        df = pd.read_csv(Config.TRAIN_CSV)
        df['file_path'] = df['id_code'].apply(lambda x: os.path.join(Config.IMAGE_DIR, f"{x}.png"))
        df = df[df['file_path'].apply(os.path.exists)]
        
        logger.info(f"Total images: {len(df)}")
        
        # Original distribution
        logger.info("\nOriginal distribution:")
        for cls in range(5):
            count = (df['diagnosis'] == cls).sum()
            logger.info(f"  {Config.CLASS_NAMES[cls]:15}: {count:4} ({100*count/len(df):5.1f}%)")
        
        # Stratified split
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=Config.SEED, stratify=df['diagnosis']
        )
        
        # Oversampling
        max_count = train_df['diagnosis'].value_counts().max()
        target_count = int(max_count * 0.85)
        
        balanced = []
        for cls in range(5):
            cls_df = train_df[train_df['diagnosis'] == cls]
            if len(cls_df) < target_count:
                oversampled = cls_df.sample(n=target_count, replace=True, random_state=Config.SEED)
                balanced.append(oversampled)
            else:
                balanced.append(cls_df)
        
        train_df = pd.concat(balanced, ignore_index=True).sample(frac=1, random_state=Config.SEED)
        
        logger.info(f"\nAfter oversampling:")
        logger.info(f"  Training samples: {len(train_df)}")
        logger.info(f"  Validation samples: {len(val_df)}")
        
        # Class weights
        counts = np.bincount(train_df['diagnosis'].values, minlength=5)
        total = counts.sum()
        class_weights = {i: total / (5 * c + 1) for i, c in enumerate(counts)}
        
        return train_df, val_df, class_weights
    
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise DataLoadError(f"Data loading failed: {e}", sys.exc_info())


def create_dataset_v2s(df, augment=False):
    """Create dataset for EfficientNetV2-S."""
    def load_fn(path, label):
        def _load(p, l):
            img = preprocess_image_v2s(p.numpy().decode('utf-8'), Config.IMAGE_SIZE)
            return img.astype(np.float32), np.int32(l)
        
        img, lab = tf.py_function(_load, [path, label], [tf.float32, tf.int32])
        img.set_shape((*Config.IMAGE_SIZE, 3))
        lab.set_shape([])
        return img, tf.one_hot(lab, depth=Config.NUM_CLASSES)
    
    ds = tf.data.Dataset.from_tensor_slices((df['file_path'].values, df['diagnosis'].values))
    
    if augment:
        ds = ds.shuffle(len(df), seed=Config.SEED)
    
    ds = ds.map(load_fn, num_parallel_calls=AUTOTUNE)
    
    if augment:
        aug_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        ds = ds.batch(Config.BATCH_SIZE, drop_remainder=True)
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.batch(Config.BATCH_SIZE)
    
    return ds.prefetch(AUTOTUNE)


def create_dataset_b_series(df, augment=False):
    """Create dataset for EfficientNetB0/B2."""
    def load_fn(path, label):
        def _load(p, l):
            img = preprocess_image_b_series(p.numpy().decode('utf-8'), Config.IMAGE_SIZE)
            return img.astype(np.float32), np.int32(l)
        
        img, lab = tf.py_function(_load, [path, label], [tf.float32, tf.int32])
        img.set_shape((*Config.IMAGE_SIZE, 3))
        lab.set_shape([])
        return img, tf.one_hot(lab, depth=Config.NUM_CLASSES)
    
    ds = tf.data.Dataset.from_tensor_slices((df['file_path'].values, df['diagnosis'].values))
    
    if augment:
        ds = ds.shuffle(len(df), seed=Config.SEED)
    
    ds = ds.map(load_fn, num_parallel_calls=AUTOTUNE)
    
    if augment:
        aug_layer = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        ds = ds.batch(Config.BATCH_SIZE, drop_remainder=True)
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.batch(Config.BATCH_SIZE)
    
    return ds.prefetch(AUTOTUNE)


# =====================================================
# DEEP MONITORING CALLBACK
# =====================================================
class DeepMetricsCallback(tf.keras.callbacks.Callback):
    """Deep parameter monitoring with comprehensive metrics per epoch."""
    
    def __init__(self, val_dataset, model_name, log_dir):
        super().__init__()
        self.val_dataset = val_dataset
        self.model_name = model_name
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, model_name))
        
        self.history = {
            'epoch': [], 'loss': [], 'val_loss': [],
            'accuracy': [], 'val_accuracy': [],
            'qwk': [], 'auc': [], 'f1': [],
            'per_class_acc': [], 'lr': []
        }
        
        self.best_qwk = -1
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        try:
            # Collect predictions
            y_true, y_pred_proba = [], []
            for batch_x, batch_y in self.val_dataset:
                preds = self.model.predict(batch_x, verbose=0)
                y_pred_proba.extend(preds)
                y_true.extend(np.argmax(batch_y.numpy(), axis=-1))
            
            y_true = np.array(y_true)
            y_pred_proba = np.array(y_pred_proba)
            y_pred = np.argmax(y_pred_proba, axis=-1)
            confidences = np.max(y_pred_proba, axis=1)
            
            # Calculate metrics
            qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            f1 = f1_score(y_true, y_pred, average='macro')
            acc = accuracy_score(y_true, y_pred)
            
            try:
                y_true_oh = np.eye(5)[y_true]
                auc_score = roc_auc_score(y_true_oh, y_pred_proba, multi_class='ovr', average='macro')
            except:
                auc_score = 0
            
            # Per-class accuracy
            cm = confusion_matrix(y_true, y_pred, labels=range(5))
            per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
            
            # Get learning rate
            try:
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            except:
                lr = 0
            
            # Store history
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(float(logs.get('loss', 0)))
            self.history['val_loss'].append(float(logs.get('val_loss', 0)))
            self.history['accuracy'].append(float(logs.get('accuracy', 0)))
            self.history['val_accuracy'].append(float(logs.get('val_accuracy', 0)))
            self.history['qwk'].append(qwk)
            self.history['auc'].append(auc_score)
            self.history['f1'].append(f1)
            self.history['per_class_acc'].append(per_class_acc.tolist())
            self.history['lr'].append(lr)
            
            # TensorBoard logging
            with self.writer.as_default():
                tf.summary.scalar('metrics/qwk', qwk, step=epoch)
                tf.summary.scalar('metrics/auc', auc_score, step=epoch)
                tf.summary.scalar('metrics/f1', f1, step=epoch)
                tf.summary.scalar('metrics/accuracy', acc, step=epoch)
                tf.summary.scalar('training/loss', logs.get('loss', 0), step=epoch)
                tf.summary.scalar('training/val_loss', logs.get('val_loss', 0), step=epoch)
                tf.summary.scalar('training/learning_rate', lr, step=epoch)
                
                for i, cls_acc in enumerate(per_class_acc):
                    tf.summary.scalar(f'per_class/{Config.CLASS_NAMES[i]}', cls_acc, step=epoch)
            
            # Print detailed metrics
            logger.info(f"\n  [{self.model_name.upper()}] Epoch {epoch+1}:")
            logger.info(f"    Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
            logger.info(f"    Acc:  {logs.get('accuracy', 0):.4f} | Val Acc: {logs.get('val_accuracy', 0):.4f}")
            logger.info(f"    QWK:  {qwk:.4f} | AUC: {auc_score:.4f} | F1: {f1:.4f}")
            logger.info(f"    Per-class: " + " | ".join([f"{Config.CLASS_NAMES[i][:3]}:{a:.2f}" for i, a in enumerate(per_class_acc)]))
            
            # Track best
            if qwk > self.best_qwk:
                self.best_qwk = qwk
                self.best_epoch = epoch + 1
                logger.info(f"    *** NEW BEST QWK: {qwk:.4f} ***")
        
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}")
    
    def get_summary(self):
        return {
            'best_qwk': self.best_qwk,
            'best_epoch': self.best_epoch,
            'final_qwk': self.history['qwk'][-1] if self.history['qwk'] else 0,
            'final_auc': self.history['auc'][-1] if self.history['auc'] else 0,
            'final_f1': self.history['f1'][-1] if self.history['f1'] else 0,
            'final_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0,
            'final_accuracy': self.history['val_accuracy'][-1] if self.history['val_accuracy'] else 0,
            'history': self.history
        }


# =====================================================
# MODEL BUILDERS - EfficientNet-V2S, B2, B0
# =====================================================
def build_efficientnet_v2s():
    """Build EfficientNetV2-S model."""
    logger.info("Building EfficientNetV2-S...")
    tf.keras.backend.clear_session()
    gc.collect()
    
    inputs = tf.keras.Input(shape=(*Config.IMAGE_SIZE, 3), name='input')
    base = tf.keras.applications.EfficientNetV2S(
        include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg'
    )
    base.trainable = False
    
    x = base.output
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name='efficientnet_v2s')
    logger.info(f"  EfficientNetV2-S built: {model.count_params():,} parameters")
    return model, base


def build_efficientnet_b2():
    """Build EfficientNetB2 model."""
    logger.info("Building EfficientNetB2...")
    tf.keras.backend.clear_session()
    gc.collect()
    
    inputs = tf.keras.Input(shape=(*Config.IMAGE_SIZE, 3), name='input')
    base = tf.keras.applications.EfficientNetB2(
        include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg'
    )
    base.trainable = False
    
    x = base.output
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name='efficientnet_b2')
    logger.info(f"  EfficientNetB2 built: {model.count_params():,} parameters")
    return model, base


def build_efficientnet_b0():
    """Build EfficientNetB0 model."""
    logger.info("Building EfficientNetB0...")
    tf.keras.backend.clear_session()
    gc.collect()
    
    inputs = tf.keras.Input(shape=(*Config.IMAGE_SIZE, 3), name='input')
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg'
    )
    base.trainable = False
    
    x = base.output
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name='efficientnet_b0')
    logger.info(f"  EfficientNetB0 built: {model.count_params():,} parameters")
    return model, base


# =====================================================
# TRAINING
# =====================================================
def train_single_model(model, base_model, train_ds, val_ds, class_weights, model_name, log_dir):
    """Train a single model with deep monitoring."""
    try:
        logger.info("=" * 60)
        logger.info(f"TRAINING: {model_name.upper()}")
        logger.info("=" * 60)
        
        monitor = DeepMetricsCallback(val_ds, model_name, log_dir)
        checkpoint_path = os.path.join(Config.MODEL_DIR, f'{model_name}_best.weights.h5')
        
        # Phase 1: Train head
        logger.info("\nPhase 1: Training classifier head...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.HEAD_LR),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        model.fit(
            train_ds, epochs=Config.HEAD_EPOCHS, validation_data=val_ds,
            class_weight=class_weights, verbose=1,
            callbacks=[
                monitor,
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
            ]
        )
        
        # Phase 2: Fine-tune
        logger.info("\nPhase 2: Fine-tuning top layers...")
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.FINETUNE_LR),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        model.fit(
            train_ds, epochs=Config.FINETUNE_EPOCHS, validation_data=val_ds,
            class_weight=class_weights, verbose=1,
            callbacks=[
                monitor,
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True
                )
            ]
        )
        
        # Save final weights
        final_path = os.path.join(Config.MODEL_DIR, f'{model_name}_final.weights.h5')
        model.save_weights(final_path)
        logger.info(f"  Model saved: {final_path}")
        
        summary = monitor.get_summary()
        logger.info(f"\n  {model_name} Final Results:")
        logger.info(f"    Best QWK: {summary['best_qwk']:.4f} (Epoch {summary['best_epoch']})")
        logger.info(f"    Final: QWK={summary['final_qwk']:.4f}, AUC={summary['final_auc']:.4f}, F1={summary['final_f1']:.4f}")
        
        return model, summary
    
    except Exception as e:
        logger.error(f"Training {model_name} failed: {e}")
        raise


# =====================================================
# ENSEMBLE EVALUATION
# =====================================================
def evaluate_ensemble(models, weights, val_ds):
    """Evaluate weighted ensemble."""
    try:
        logger.info("=" * 60)
        logger.info("EVALUATING ENSEMBLE")
        logger.info("=" * 60)
        
        y_true, y_pred_proba = [], []
        
        for batch_x, batch_y in val_ds:
            ensemble_preds = np.zeros((batch_x.shape[0], Config.NUM_CLASSES))
            
            for model, weight in zip(models, weights):
                preds = model.predict(batch_x, verbose=0)
                ensemble_preds += preds * weight
            
            y_pred_proba.extend(ensemble_preds)
            y_true.extend(np.argmax(batch_y.numpy(), axis=-1))
        
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        y_pred = np.argmax(y_pred_proba, axis=-1)
        
        # Calculate all metrics
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        f1 = f1_score(y_true, y_pred, average='macro')
        acc = accuracy_score(y_true, y_pred)
        
        try:
            y_true_oh = np.eye(5)[y_true]
            auc = roc_auc_score(y_true_oh, y_pred_proba, multi_class='ovr', average='macro')
        except:
            auc = 0
        
        # Per-class metrics
        cm = confusion_matrix(y_true, y_pred, labels=range(5))
        per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
        
        return {
            'qwk': qwk, 'auc': auc, 'f1': f1, 'accuracy': acc,
            'per_class_acc': per_class_acc.tolist(),
            'confusion_matrix': cm.tolist()
        }, y_true, y_pred, y_pred_proba
    
    except Exception as e:
        logger.error(f"Ensemble evaluation failed: {e}")
        raise


# =====================================================
# VISUALIZATION
# =====================================================
def generate_graphs(results, y_true, y_pred, output_dir):
    """Generate comprehensive visualization graphs."""
    try:
        logger.info("Generating visualization graphs...")
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES)
        plt.title('Ensemble Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        logger.info(f"  Graphs saved to: {output_dir}")
    
    except Exception as e:
        logger.warning(f"Graph generation failed: {e}")


# =====================================================
# MAIN
# =====================================================
def train_efficientnet_ensemble():
    """Train industry-grade EfficientNet ensemble."""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(Config.LOG_DIR, timestamp)
        os.makedirs(log_dir, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("EFFICIENTNET ENSEMBLE TRAINING")
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info(f"Target QWK: {Config.TARGET_QWK}")
        logger.info(f"Target AUC: {Config.TARGET_AUC}")
        logger.info(f"Models: EfficientNetV2-S, EfficientNetB2, EfficientNetB0")
        logger.info(f"Image size: {Config.IMAGE_SIZE}")
        logger.info(f"Batch size: {Config.BATCH_SIZE}")
        
        # Load data
        train_df, val_df, class_weights = load_data()
        
        results = {'model_results': {}}
        trained_models = []
        
        # Train EfficientNetV2-S (uses V2 preprocessing)
        train_ds_v2s = create_dataset_v2s(train_df, augment=True)
        val_ds_v2s = create_dataset_v2s(val_df, augment=False)
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL 1/3: EfficientNetV2-S")
        logger.info("=" * 60)
        model1, base1 = build_efficientnet_v2s()
        model1, summary1 = train_single_model(model1, base1, train_ds_v2s, val_ds_v2s, class_weights, 'efficientnet_v2s', log_dir)
        results['model_results']['efficientnet_v2s'] = summary1
        trained_models.append(model1)
        
        gc.collect()
        
        # Train EfficientNetB2 (uses B-series preprocessing)
        train_ds_b = create_dataset_b_series(train_df, augment=True)
        val_ds_b = create_dataset_b_series(val_df, augment=False)
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL 2/3: EfficientNetB2")
        logger.info("=" * 60)
        model2, base2 = build_efficientnet_b2()
        model2, summary2 = train_single_model(model2, base2, train_ds_b, val_ds_b, class_weights, 'efficientnet_b2', log_dir)
        results['model_results']['efficientnet_b2'] = summary2
        trained_models.append(model2)
        
        gc.collect()
        
        # Train EfficientNetB0 (uses B-series preprocessing)
        train_ds_b = create_dataset_b_series(train_df, augment=True)
        val_ds_b = create_dataset_b_series(val_df, augment=False)
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL 3/3: EfficientNetB0")
        logger.info("=" * 60)
        model3, base3 = build_efficientnet_b0()
        model3, summary3 = train_single_model(model3, base3, train_ds_b, val_ds_b, class_weights, 'efficientnet_b0', log_dir)
        results['model_results']['efficientnet_b0'] = summary3
        trained_models.append(model3)
        
        # Create weighted ensemble based on QWK
        qwk_scores = [summary1['best_qwk'], summary2['best_qwk'], summary3['best_qwk']]
        total_qwk = sum(qwk_scores) + 1e-8
        weights = [q / total_qwk for q in qwk_scores]
        
        logger.info("\n" + "=" * 60)
        logger.info("ENSEMBLE WEIGHTS")
        logger.info("=" * 60)
        logger.info(f"  EfficientNetV2-S: {weights[0]:.3f} (QWK: {qwk_scores[0]:.4f})")
        logger.info(f"  EfficientNetB2:   {weights[1]:.3f} (QWK: {qwk_scores[1]:.4f})")
        logger.info(f"  EfficientNetB0:   {weights[2]:.3f} (QWK: {qwk_scores[2]:.4f})")
        
        # Evaluate ensemble using B-series preprocessing (as majority uses this)
        val_ds_b = create_dataset_b_series(val_df, augment=False)
        ensemble_results, y_true, y_pred, y_pred_proba = evaluate_ensemble(trained_models, weights, val_ds_b)
        results['ensemble'] = ensemble_results
        results['weights'] = weights
        
        # Final results
        logger.info("\n" + "=" * 80)
        logger.info("FINAL ENSEMBLE RESULTS")
        logger.info("=" * 80)
        logger.info(f"  QWK:      {ensemble_results['qwk']:.4f} (Target: {Config.TARGET_QWK})")
        logger.info(f"  AUC:      {ensemble_results['auc']:.4f} (Target: {Config.TARGET_AUC})")
        logger.info(f"  F1:       {ensemble_results['f1']:.4f}")
        logger.info(f"  Accuracy: {ensemble_results['accuracy']:.4f}")
        
        if ensemble_results['qwk'] >= Config.TARGET_QWK:
            logger.info("\n  *** TARGET QWK ACHIEVED! ***")
        
        logger.info("\nPer-class Accuracy:")
        for i, (name, acc) in enumerate(zip(Config.CLASS_NAMES, ensemble_results['per_class_acc'])):
            status = "✓" if acc >= 0.55 else "✗"
            logger.info(f"  {status} {name:15}: {acc:.2%}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=Config.CLASS_NAMES))
        
        # Generate graphs
        generate_graphs(results, y_true, y_pred, Config.GRAPH_DIR)
        
        # Save results
        results_path = os.path.join(Config.OUTPUT_DIR, 'ensemble_results.json')
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        with open(results_path, 'w') as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Models saved: {Config.MODEL_DIR}")
        logger.info(f"Graphs saved: {Config.GRAPH_DIR}")
        logger.info(f"Results: {results_path}")
        logger.info(f"TensorBoard: tensorboard --logdir {log_dir}")
        logger.info("=" * 80)
        
        return trained_models, results
    
    except Exception as e:
        logger.critical(f"Training failed: {e}")
        raise CustomException(f"Training failed: {e}", sys.exc_info())


if __name__ == "__main__":
    try:
        logger.info("Starting EfficientNet Ensemble Training...")
        models, results = train_efficientnet_ensemble()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
