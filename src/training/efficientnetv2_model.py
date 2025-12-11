"""
src/training/efficientnetv2_model.py

EfficientNetV2-M model for Diabetic Retinopathy classification.
Features:
- EfficientNetV2-M backbone (pretrained on ImageNet21k or ImageNet1k)
- Attention-based classification head
- Support for ordinal regression output
- Progressive fine-tuning
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2M, EfficientNetV2S, EfficientNetV2L


def squeeze_excite_block(input_tensor, reduction=16, name_prefix='se'):
    """Squeeze-and-Excitation attention block."""
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D(name=f'{name_prefix}_gap')(input_tensor)
    se = layers.Dense(filters // reduction, activation='relu', name=f'{name_prefix}_dense1')(se)
    se = layers.Dense(filters, activation='sigmoid', name=f'{name_prefix}_dense2')(se)
    se = layers.Reshape((1, 1, filters), name=f'{name_prefix}_reshape')(se)
    return layers.Multiply(name=f'{name_prefix}_mul')([input_tensor, se])


def build_efficientnetv2_model(
    input_shape=(640, 640, 3),
    num_classes=5,
    model_size='m',  # 's', 'm', or 'l'
    pretrained_weights='imagenet',
    freeze_base=True,
    trainable_layers=0,
    dropout_rate=0.5,
    use_attention=True,
    use_ordinal_output=False,
    dense_units=[512, 256],
    l2_reg=1e-4
):
    """
    Build EfficientNetV2 model for DR classification.
    
    Args:
        input_shape: Input image shape (H, W, C)
        num_classes: Number of output classes
        model_size: 's' (small), 'm' (medium), or 'l' (large)
        pretrained_weights: 'imagenet' or None
        freeze_base: Whether to freeze base model initially
        trainable_layers: Number of layers to make trainable from top (if freeze_base=False)
        dropout_rate: Dropout rate for classification head
        use_attention: Whether to use SE attention block
        use_ordinal_output: Whether to add ordinal regression head (CORAL)
        dense_units: List of dense layer units
        l2_reg: L2 regularization strength
    
    Returns:
        Keras Model
    """
    
    print(f"\n{'='*60}")
    print(f"Building EfficientNetV2-{model_size.upper()}")
    print(f"  Input shape: {input_shape}")
    print(f"  Classes: {num_classes}")
    print(f"  Weights: {pretrained_weights}")
    print(f"  Trainable layers: {'All' if not freeze_base else trainable_layers}")
    print(f"{'='*60}")
    
    # Input
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Select EfficientNetV2 variant
    if model_size == 's':
        base_model = EfficientNetV2S(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=inputs,
            pooling=None
        )
    elif model_size == 'm':
        base_model = EfficientNetV2M(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=inputs,
            pooling=None
        )
    else:  # 'l'
        base_model = EfficientNetV2L(
            include_top=False,
            weights=pretrained_weights,
            input_tensor=inputs,
            pooling=None
        )
    
    # Freeze/unfreeze base model
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
    
    # Get feature maps
    x = base_model.output
    
    # Optional attention block
    if use_attention:
        x = squeeze_excite_block(x, reduction=16, name_prefix='head_se')
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='head_global_pool')(x)
    
    # Classification head
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    
    for i, units in enumerate(dense_units):
        x = layers.Dense(
            units, 
            activation='relu', 
            kernel_regularizer=regularizer,
            name=f'head_dense_{i+1}'
        )(x)
        x = layers.BatchNormalization(name=f'head_bn_{i+1}')(x)
        x = layers.Dropout(dropout_rate * (0.8 ** i), name=f'head_dropout_{i+1}')(x)
    
    # Main classification output (softmax)
    main_output = layers.Dense(
        num_classes, 
        activation='softmax',
        kernel_regularizer=regularizer,
        name='head_classification'
    )(x)
    
    outputs = [main_output]
    
    # Optional ordinal regression output (CORAL)
    if use_ordinal_output:
        ordinal_output = layers.Dense(
            num_classes - 1,  # K-1 binary classifiers
            activation='sigmoid',
            kernel_regularizer=regularizer,
            name='ordinal'
        )(x)
        outputs.append(ordinal_output)
    
    # Build model
    model = Model(
        inputs=inputs,
        outputs=outputs if len(outputs) > 1 else outputs[0],
        name=f'efficientnetv2_{model_size}_dr'
    )
    
    # Print summary
    trainable_count = sum([tf.reduce_prod(w.shape) for w in model.trainable_weights])
    non_trainable_count = sum([tf.reduce_prod(w.shape) for w in model.non_trainable_weights])
    
    print(f"\nModel built!")
    print(f"  Total params: {(trainable_count + non_trainable_count):,}")
    print(f"  Trainable params: {trainable_count:,}")
    print(f"  Non-trainable params: {non_trainable_count:,}")
    print(f"{'='*60}\n")
    
    return model


def unfreeze_model(model, layers_to_unfreeze=100, learning_rate=1e-5):
    """
    Unfreeze top layers of the model for fine-tuning.
    
    Args:
        model: Keras model
        layers_to_unfreeze: Number of layers to unfreeze from the top
        learning_rate: New learning rate for fine-tuning
    
    Returns:
        Modified model
    """
    # Find base model
    base_model = None
    for layer in model.layers:
        if 'efficientnetv2' in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        # Base model is integrated, unfreeze model layers directly
        for layer in model.layers[:-layers_to_unfreeze]:
            layer.trainable = False
        for layer in model.layers[-layers_to_unfreeze:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    else:
        # Unfreeze top layers of base model
        for layer in base_model.layers[:-layers_to_unfreeze]:
            layer.trainable = False
        for layer in base_model.layers[-layers_to_unfreeze:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
    
    # Keep batch norm layers frozen (important for transfer learning)
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    
    trainable_count = sum([tf.reduce_prod(w.shape) for w in model.trainable_weights])
    print(f"Unfroze {layers_to_unfreeze} layers. Trainable params: {trainable_count:,}")
    
    return model


def create_lr_scheduler(
    initial_lr=1e-4,
    final_lr=1e-6,
    warmup_epochs=5,
    total_epochs=50,
    warmup_start_lr=1e-6,
    schedule_type='cosine'
):
    """
    Create a learning rate scheduler with warmup.
    
    Args:
        initial_lr: Peak learning rate after warmup
        final_lr: Minimum learning rate
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        warmup_start_lr: Starting LR for warmup
        schedule_type: 'cosine', 'cosine_restarts', or 'exponential'
    
    Returns:
        Learning rate schedule function
    """
    
    def warmup_cosine_schedule(epoch, lr):
        if epoch < warmup_epochs:
            # Linear warmup
            return warmup_start_lr + (initial_lr - warmup_start_lr) * (epoch / warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return final_lr + 0.5 * (initial_lr - final_lr) * (1 + tf.math.cos(progress * 3.14159))
    
    def warmup_cosine_restarts_schedule(epoch, lr):
        if epoch < warmup_epochs:
            return warmup_start_lr + (initial_lr - warmup_start_lr) * (epoch / warmup_epochs)
        else:
            # Cosine with restarts every 10 epochs
            restart_period = 10
            cycle = ((epoch - warmup_epochs) // restart_period)
            cycle_epoch = (epoch - warmup_epochs) % restart_period
            cycle_lr = initial_lr * (0.8 ** cycle)  # Decay max LR each cycle
            progress = cycle_epoch / restart_period
            return final_lr + 0.5 * (cycle_lr - final_lr) * (1 + tf.math.cos(progress * 3.14159))
    
    def warmup_exponential_schedule(epoch, lr):
        if epoch < warmup_epochs:
            return warmup_start_lr + (initial_lr - warmup_start_lr) * (epoch / warmup_epochs)
        else:
            decay_rate = (final_lr / initial_lr) ** (1 / (total_epochs - warmup_epochs))
            return initial_lr * (decay_rate ** (epoch - warmup_epochs))
    
    if schedule_type == 'cosine':
        return warmup_cosine_schedule
    elif schedule_type == 'cosine_restarts':
        return warmup_cosine_restarts_schedule
    else:
        return warmup_exponential_schedule


class TestTimeAugmentation:
    """
    Test-Time Augmentation for improved inference.
    Applies multiple augmentations and averages predictions.
    """
    
    def __init__(self, model, n_augments=5):
        self.model = model
        self.n_augments = n_augments
    
    def augment(self, image):
        """Apply random augmentation to image."""
        augmented = [image]
        
        # Horizontal flip
        augmented.append(tf.image.flip_left_right(image))
        
        # Rotations
        for k in [1, 2, 3]:
            augmented.append(tf.image.rot90(image, k=k))
        
        return augmented[:self.n_augments]
    
    def predict(self, images):
        """Predict with TTA."""
        batch_preds = []
        
        for img in images:
            aug_imgs = self.augment(img)
            aug_batch = tf.stack(aug_imgs, axis=0)
            preds = self.model.predict(aug_batch, verbose=0)
            
            # Average predictions
            avg_pred = tf.reduce_mean(preds, axis=0)
            batch_preds.append(avg_pred)
        
        return tf.stack(batch_preds, axis=0)
    
    def predict_dataset(self, dataset, batch_size=8):
        """Predict on entire dataset with TTA."""
        all_preds = []
        
        for batch in dataset:
            batch_preds = []
            for img in batch:
                aug_imgs = self.augment(img)
                aug_batch = tf.stack(aug_imgs, axis=0)
                preds = self.model.predict(aug_batch, verbose=0)
                avg_pred = tf.reduce_mean(preds, axis=0)
                batch_preds.append(avg_pred)
            
            all_preds.extend(batch_preds)
        
        return tf.stack(all_preds, axis=0)


if __name__ == "__main__":
    # Test model creation
    model = build_efficientnetv2_model(
        input_shape=(640, 640, 3),
        num_classes=5,
        model_size='m',
        pretrained_weights='imagenet',
        freeze_base=True,
        use_attention=True
    )
    
    model.summary(show_trainable=True)
