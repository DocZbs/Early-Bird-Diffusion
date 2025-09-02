"""
Simplified Timestep Importance Sampling for UNet-based Diffusion Training

This implements a clean, simple timestep importance sampling approach that:
1. Tracks loss history per timestep
2. Dynamically adjusts sampling probabilities based on losses
3. Uses EMA for smooth probability updates
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict


class TimestepImportanceSampler:
    """
    Simple timestep importance sampling based on loss history.
    """
    
    def __init__(self, num_timesteps, device, 
                 ema_decay=0.99, 
                 min_prob=0.1, 
                 warmup_steps=1000,
                 update_freq=100):
        """
        Args:
            num_timesteps: Total number of diffusion timesteps (T)
            device: Torch device
            ema_decay: EMA decay rate for loss tracking
            min_prob: Minimum sampling probability per timestep
            warmup_steps: Steps to use uniform sampling before importance sampling
            update_freq: How often to update sampling probabilities
        """
        self.num_timesteps = num_timesteps
        self.device = device
        self.ema_decay = ema_decay
        self.min_prob = min_prob
        self.warmup_steps = warmup_steps
        self.update_freq = update_freq
        
        # Loss tracking per timestep
        self.loss_history = torch.ones(num_timesteps, device=device)
        self.loss_counts = torch.zeros(num_timesteps, device=device)
        
        # Sampling probabilities (uniform initially)
        self.probs = torch.ones(num_timesteps, device=device) / num_timesteps
        
        self.step_count = 0
        
        logging.info(f"Timestep Importance Sampler initialized with {num_timesteps} timesteps")
    
    def sample_timesteps(self, batch_size):
        """
        Sample timesteps according to current importance probabilities.
        
        Args:
            batch_size: Number of timesteps to sample
            
        Returns:
            timesteps: Sampled timesteps [batch_size]
            weights: Importance sampling weights [batch_size]
        """
        if self.step_count < self.warmup_steps:
            # Uniform sampling during warmup
            timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
            weights = torch.ones(batch_size, device=self.device)
        else:
            # Importance sampling
            timesteps = torch.multinomial(self.probs, batch_size, replacement=True)
            
            # Compute importance weights: 1 / (batch_size * prob)
            sampled_probs = self.probs[timesteps]
            weights = 1.0 / (sampled_probs * batch_size)
        
        return timesteps, weights
    
    def update_loss_history(self, timesteps, losses):
        """
        Update loss history with new observations.
        
        Args:
            timesteps: Timesteps that were used [batch_size]
            losses: Per-sample losses [batch_size]
        """
        if not torch.is_tensor(losses):
            losses = torch.tensor(losses, device=self.device)
        
        if losses.dim() > 1:
            # If losses have extra dimensions, take mean over non-batch dims
            losses = losses.flatten(1).mean(1)
        
        # Update EMA loss estimates
        for i, (t, loss) in enumerate(zip(timesteps, losses)):
            t_idx = t.item() if torch.is_tensor(t) else t
            if t_idx < 0 or t_idx >= self.num_timesteps:
                continue
                
            if self.loss_counts[t_idx] == 0:
                # First observation for this timestep
                self.loss_history[t_idx] = loss.item()
            else:
                # EMA update
                self.loss_history[t_idx] = (
                    self.ema_decay * self.loss_history[t_idx] + 
                    (1 - self.ema_decay) * loss.item()
                )
            
            self.loss_counts[t_idx] += 1
    
    def update_probabilities(self):
        """
        Update sampling probabilities based on current loss history.
        Higher losses get higher probabilities.
        """
        if self.step_count < self.warmup_steps:
            return
        
        # Convert losses to probabilities (higher loss = higher prob)
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        raw_probs = self.loss_history + eps
        
        # Normalize to get probabilities
        raw_probs = raw_probs / raw_probs.sum()
        
        # Apply minimum probability constraint
        self.probs = torch.clamp(raw_probs, min=self.min_prob / self.num_timesteps)
        self.probs = self.probs / self.probs.sum()  # Renormalize
        
        if self.step_count % (self.update_freq * 10) == 0:
            # Log statistics periodically
            max_prob_idx = torch.argmax(self.probs)
            min_prob_idx = torch.argmin(self.probs)
            logging.info(
                f"Timestep sampling - Max prob: t={max_prob_idx} ({self.probs[max_prob_idx]:.4f}), "
                f"Min prob: t={min_prob_idx} ({self.probs[min_prob_idx]:.4f}), "
                f"Entropy: {(-self.probs * torch.log(self.probs + 1e-8)).sum():.4f}"
            )
    
    def step(self):
        """Step counter and periodic updates."""
        self.step_count += 1
        
        # Update probabilities periodically
        if self.step_count % self.update_freq == 0:
            self.update_probabilities()
    
    def get_stats(self):
        """Get statistics for logging."""
        stats = {
            'step': self.step_count,
            'warmup_phase': self.step_count < self.warmup_steps,
            'loss_mean': self.loss_history.mean().item(),
            'loss_std': self.loss_history.std().item(),
            'prob_entropy': (-self.probs * torch.log(self.probs + 1e-8)).sum().item(),
            'prob_max': self.probs.max().item(),
            'prob_min': self.probs.min().item(),
        }
        
        # Add top-5 most important timesteps
        top_indices = torch.topk(self.probs, 5).indices
        for i, idx in enumerate(top_indices):
            stats[f'top{i+1}_timestep'] = idx.item()
            stats[f'top{i+1}_prob'] = self.probs[idx].item()
        
        return stats
    
    def get_timestep_weights(self):
        """Get current timestep sampling probabilities."""
        return self.probs.clone()
    
    def reset_history(self):
        """Reset loss history (useful for training restarts)."""
        self.loss_history = torch.ones(self.num_timesteps, device=self.device)
        self.loss_counts = torch.zeros(self.num_timesteps, device=self.device)
        self.probs = torch.ones(self.num_timesteps, device=self.device) / self.num_timesteps
        logging.info("Reset timestep importance sampling history")