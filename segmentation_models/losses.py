from . import inject_global_losses
from . import inject_global_submodules
from .metrics import jaccard_score, f_score

SMOOTH = 1.

__all__ = [
    'jaccard_loss', 'bce_jaccard_loss', 'cce_jaccard_loss',
    'dice_loss', 'bce_dice_loss', 'cce_dice_loss',
]


# ============================== Jaccard Losses ==============================

def jaccard_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True):
    r"""Jaccard loss function for imbalanced datasets:

    .. math:: L(A, B) = 1 - \frac{A \cap B}{A \cup B}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        Jaccard loss in range [0, 1]

    """
    return 1 - jaccard_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image)


@inject_global_losses
@inject_global_submodules
def bce_jaccard_loss(gt, pr, bce_weight=1., smooth=SMOOTH, per_image=True, **kwargs):
    r"""Sum of binary crossentropy and jaccard losses:
    
    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + jaccard_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for jaccard loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, jaccard loss is calculated as mean over images in batch (B),
            else over whole batch (only for jaccard loss)

    Returns:
        loss
    
    """
    backend = kwargs['backend']
    losses = kwargs['losses']

    bce = backend.mean(losses.binary_crossentropy(gt, pr))
    loss = bce_weight * bce + jaccard_loss(gt, pr, smooth=smooth, per_image=per_image)
    return loss


@inject_global_losses
@inject_global_submodules
def cce_jaccard_loss(gt, pr, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True, **kwargs):
    r"""Sum of categorical crossentropy and jaccard losses:
    
    .. math:: L(A, B) = cce_weight * categorical_crossentropy(A, B) + jaccard_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for jaccard loss, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, jaccard loss is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        loss
    
    """
    backend = kwargs['backend']
    losses = kwargs['losses']

    cce = losses.categorical_crossentropy(gt, pr) * class_weights
    cce = backend.mean(cce)
    return cce_weight * cce + jaccard_loss(gt, pr, smooth=smooth, class_weights=class_weights, per_image=per_image)


# # Update custom objects
# get_custom_objects().update({
#     'jaccard_loss': jaccard_loss,
#     'bce_jaccard_loss': bce_jaccard_loss,
#     'cce_jaccard_loss': cce_jaccard_loss,
# })


# ============================== Dice Losses ================================

def dice_loss(gt, pr, class_weights=1., smooth=SMOOTH, per_image=True, beta=1., **kwargs):
    r"""Dice loss function for imbalanced datasets:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        Dice loss in range [0, 1]

    """
    return 1 - f_score(gt, pr, class_weights=class_weights, smooth=smooth, per_image=per_image, beta=beta)

@inject_global_losses
@inject_global_submodules
def bce_dice_loss(gt, pr, bce_weight=1., smooth=SMOOTH, per_image=True, beta=1., **kwargs):
    r"""Sum of binary crossentropy and dice losses:
    
    .. math:: L(A, B) = bce_weight * binary_crossentropy(A, B) + dice_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C 
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss
    
    """
    backend = kwargs['backend']
    losses = kwargs['losses']

    bce = backend.mean(losses.binary_crossentropy(gt, pr))
    loss = bce_weight * bce + dice_loss(gt, pr, smooth=smooth, per_image=per_image, beta=beta)
    return loss

@inject_global_losses
@inject_global_submodules
def cce_dice_loss(gt, pr, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True, beta=1., **kwargs):
    r"""Sum of categorical crossentropy and dice losses:
    
    .. math:: L(A, B) = cce_weight * categorical_crossentropy(A, B) + dice_loss(A, B)
    
    Args:
        gt: ground truth 4D keras tensor (B, H, W, C)
        pr: prediction 4D keras tensor (B, H, W, C)
        class_weights: 1. or list of class weights for dice loss, len(weights) = C 
        smooth: value to avoid division by zero
        per_image: if ``True``, dice loss is calculated as mean over images in batch (B),
            else over whole batch
        beta: coefficient for precision recall balance

    Returns:
        loss
    
    """
    backend = kwargs['backend']
    losses = kwargs['losses']

    cce = losses.categorical_crossentropy(gt, pr) * class_weights
    cce = backend.mean(cce)
    return cce_weight * cce + dice_loss(gt, pr, smooth=smooth, class_weights=class_weights, per_image=per_image,
                                        beta=beta)


# # Update custom objects
# get_custom_objects().update({
#     'dice_loss': dice_loss,
#     'bce_dice_loss': bce_dice_loss,
#     'cce_dice_loss': cce_dice_loss,
# })
