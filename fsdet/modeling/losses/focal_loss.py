import torch
import torch.jit
import torch.nn.functional as F

def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float=-1, 
               gamma: float=2, reduction: str='mean') -> torch.Tensor:
    """
    Compute Focal Loss.
    
    Args:
        inputs (torch.Tensor): Predictions with shape [batch_size, num_classes].
        targets (torch.Tensor): Ground truth labels with shape [batch_size].
        gamma (float): Focusing parameter. Default is 2.
        alpha (float, list, torch.Tensor, optional): Balancing parameter. Default is None.
        reduction (str): Reduction method to apply to the output. Options are 'none', 'mean', 'sum'. Default is 'mean'.
        
    Returns:
        torch.Tensor: Computed Focal Loss.
    """
    
    # Compute cross-entropy loss
    BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
    
    # Compute the pt value (probability of the true class)
    pt = torch.exp(-BCE_loss)
    
    # Compute the focal loss
    F_loss = (1 - pt) ** gamma * BCE_loss
    
    # Apply class weights if provided
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        F_loss = alpha_t * F_loss
    
    # Apply reduction method
    if reduction == 'mean':
        loss = F_loss.mean()
    elif reduction == 'sum':
        loss = F_loss.sum()
    else:
        loss = F_loss
    return loss
    
focal_loss_jit: "torch.jit.ScriptModule" = torch.jit.script(focal_loss)