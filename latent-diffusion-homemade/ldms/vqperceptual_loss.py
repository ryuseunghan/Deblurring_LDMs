# vqperceptual_loss.py
import torch
import torch.nn.functional as F

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real), dim=[1,2,3])
    loss_fake = torch.mean(F.relu(1. + logits_fake), dim=[1,2,3])
    return 0.5 * (loss_real + loss_fake)

def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real), dim=[1,2,3])
    loss_fake = torch.mean(F.softplus(logits_fake), dim=[1,2,3])
    return loss_real + loss_fake
