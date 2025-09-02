"""
VR-DiT: Variance-Reduced Training for Diffusion Transformers

This module implements three variance reduction techniques:
1. NSS: Neyman Stratified Sampling for optimal timestep allocation
2. ASN: Antithetic + Sobol Noise for low-discrepancy sampling
3. mSVRG: mini Stochastic Variance Reduced Gradient with control variates
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from scipy.stats import norm
from collections import defaultdict, deque
import copy

try:
    # Try importing scipy's sobol sequence generator
    from scipy.stats.qmc import Sobol
    HAS_SOBOL = True
except ImportError:
    # Fallback to basic quasi-random if scipy not available
    HAS_SOBOL = False
    logging.warning("Scipy not available, using basic quasi-random instead of Sobol")


class VRDiTController:
    """
    VR-DiT Controller implementing variance reduction techniques for diffusion training.
    """
    
    def __init__(self, T, device, 
                 # NSS parameters
                 use_nss=True, nss_strata=4, nss_beta=0.05, nss_use_weights=False,
                 # ASN parameters  
                 use_asn=True, asn_antithetic=True, asn_sobol=True,
                 # mSVRG parameters
                 use_msvrg=True, msvrg_snapshot_freq=500, msvrg_buffer_size=256):
        """
        Initialize VR-DiT controller.
        
        Args:
            T: Number of diffusion timesteps
            device: Torch device
            use_nss: Enable Neyman Stratified Sampling
            nss_strata: Number of strata for NSS
            nss_beta: EMA decay for variance estimation
            nss_use_weights: Use importance sampling weights
            use_asn: Enable Antithetic + Sobol Noise
            asn_antithetic: Use antithetic (paired) sampling
            asn_sobol: Use Sobol low-discrepancy sequences
            use_msvrg: Enable mini-SVRG control variates
            msvrg_snapshot_freq: Snapshot frequency for mSVRG
            msvrg_buffer_size: Buffer size per stratum for mSVRG
        """
        self.T = T
        self.device = device
        
        # NSS: Neyman Stratified Sampling
        self.use_nss = use_nss
        self.nss_strata = nss_strata
        self.nss_beta = nss_beta
        self.nss_use_weights = nss_use_weights
        
        # ASN: Antithetic + Sobol Noise
        self.use_asn = use_asn
        self.asn_antithetic = asn_antithetic
        self.asn_sobol = asn_sobol
        
        # mSVRG: mini-SVRG
        self.use_msvrg = use_msvrg
        self.msvrg_snapshot_freq = msvrg_snapshot_freq
        self.msvrg_buffer_size = msvrg_buffer_size
        
        self.step_count = 0
        self._setup_nss()
        self._setup_asn()
        self._setup_msvrg()
        
    def _setup_nss(self):
        """Setup Neyman Stratified Sampling."""
        if not self.use_nss:
            return
            
        # Create strata based on SNR quantiles
        # For DDPM: SNR(t) = alpha_t^2 / sigma_t^2 = (1-beta_cum)/(beta_cum)
        # We'll approximate SNR and create strata
        t_vals = torch.arange(self.T, dtype=torch.float32)
        
        # Simple approximation of log SNR for stratification
        # This should ideally use the actual beta schedule
        log_snr_approx = -2 * torch.log(t_vals + 1) + 2 * torch.log(torch.tensor(self.T))
        
        # Create strata boundaries by quantiles
        boundaries = torch.quantile(log_snr_approx, torch.linspace(0, 1, self.nss_strata + 1))
        
        # Assign timesteps to strata
        self.strata = []
        for k in range(self.nss_strata):
            if k == self.nss_strata - 1:
                mask = (log_snr_approx >= boundaries[k])
            else:
                mask = (log_snr_approx >= boundaries[k]) & (log_snr_approx < boundaries[k+1])
            
            stratum_ts = t_vals[mask].long().tolist()
            self.strata.append(stratum_ts)
        
        # Prior probabilities (uniform for now)
        self.p_k = torch.ones(self.nss_strata, device=self.device) / self.nss_strata
        
        # Variance estimates (EMA)
        self.sigma2_k = torch.ones(self.nss_strata, device=self.device)
        
        logging.info(f"NSS: Created {self.nss_strata} strata with sizes: {[len(s) for s in self.strata]}")
        
    def _setup_asn(self):
        """Setup Antithetic + Sobol Noise."""
        if not self.use_asn:
            return
            
        # Initialize Sobol sequence generators if available
        self.sobol_initialized = False
        if self.asn_sobol and HAS_SOBOL:
            try:
                # We'll initialize the Sobol generator when we know the dimensions
                self.sobol_gen = None
                self.sobol_initialized = False
            except Exception as e:
                logging.warning(f"Failed to initialize Sobol generator: {e}")
                self.asn_sobol = False
        else:
            self.asn_sobol = False
            
    def _setup_msvrg(self):
        """Setup mini-SVRG control variates."""
        if not self.use_msvrg:
            return
            
        # Snapshot model parameters (will be set during training)
        self.snapshot_params = None
        self.snapshot_step = 0
        
        # Buffers for each stratum (store (x, t, eps) samples)
        self.buffers = [deque(maxlen=self.msvrg_buffer_size) for _ in range(self.nss_strata)]
        
        # Layer mean gradients μ_k
        self.mu_k = None  # Will be initialized when model is available
        
    def sample_timesteps_and_noise(self, batch_size, noise_shape):
        """
        Sample timesteps and noise using VR-DiT techniques.
        
        Args:
            batch_size: Batch size
            noise_shape: Shape for noise tensor (B, C, H, W)
            
        Returns:
            t: Timesteps tensor
            noise: Noise tensor  
            weights: Importance sampling weights (if using NSS weights)
        """
        if self.use_nss:
            return self._sample_nss(batch_size, noise_shape)
        else:
            return self._sample_uniform(batch_size, noise_shape)
    
    def _sample_uniform(self, batch_size, noise_shape):
        """Uniform timestep and noise sampling (baseline)."""
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        
        if self.use_asn and self.asn_antithetic:
            # Antithetic sampling for noise
            noise_half = torch.randn(batch_size // 2, *noise_shape[1:], device=self.device)
            if batch_size % 2 == 1:
                # Handle odd batch size
                extra = torch.randn(1, *noise_shape[1:], device=self.device)
                noise = torch.cat([noise_half, -noise_half, extra], dim=0)
            else:
                noise = torch.cat([noise_half, -noise_half], dim=0)
        else:
            noise = torch.randn(*noise_shape, device=self.device)
        
        weights = torch.ones(batch_size, device=self.device)
        return t, noise, weights
    
    def _sample_nss(self, batch_size, noise_shape):
        """Neyman Stratified Sampling for timesteps."""
        # Compute optimal allocation using current variance estimates
        sigma_k = torch.sqrt(torch.clamp(self.sigma2_k, min=1e-12))
        alloc = self.p_k * sigma_k
        alloc = alloc / alloc.sum()
        
        # Allocate samples to strata
        n_k = (batch_size * alloc).floor().long()
        
        # Distribute remaining samples to maintain batch_size
        remaining = batch_size - n_k.sum().item()
        if remaining > 0:
            # Add remaining samples to highest allocation strata
            _, top_indices = torch.topk(alloc, remaining)
            n_k[top_indices] += 1
        
        # Sample from each stratum
        t_list = []
        noise_list = []
        weights_list = []
        
        for k in range(self.nss_strata):
            if n_k[k] == 0:
                continue
                
            # Sample timesteps from this stratum
            if len(self.strata[k]) == 0:
                continue
                
            if self.asn_sobol and len(self.strata[k]) > 1:
                # Use low-discrepancy sampling within stratum
                t_k = self._sobol_choice(self.strata[k], n_k[k].item())
            else:
                # Uniform sampling within stratum
                indices = torch.randint(0, len(self.strata[k]), (n_k[k],), device=self.device)
                t_k = torch.tensor([self.strata[k][i] for i in indices], device=self.device)
            
            t_list.append(t_k)
            
            # Sample noise for this stratum
            if self.use_asn and self.asn_antithetic and n_k[k] > 1:
                # Antithetic noise within stratum
                noise_half = torch.randn(n_k[k] // 2, *noise_shape[1:], device=self.device)
                if n_k[k] % 2 == 1:
                    extra = torch.randn(1, *noise_shape[1:], device=self.device)
                    noise_k = torch.cat([noise_half, -noise_half, extra], dim=0)
                else:
                    noise_k = torch.cat([noise_half, -noise_half], dim=0)
            else:
                noise_k = torch.randn(n_k[k], *noise_shape[1:], device=self.device)
            
            noise_list.append(noise_k)
            
            # Compute importance weights if needed
            if self.nss_use_weights:
                # Weight = true_prob / sampling_prob = p_k / (n_k / batch_size)
                weight_k = self.p_k[k] / (n_k[k].float() / batch_size)
                weights_k = torch.full((n_k[k],), weight_k, device=self.device)
            else:
                weights_k = torch.ones(n_k[k], device=self.device)
            
            weights_list.append(weights_k)
        
        if len(t_list) == 0:
            # Fallback to uniform sampling
            return self._sample_uniform(batch_size, noise_shape)
        
        t = torch.cat(t_list)
        noise = torch.cat(noise_list)
        weights = torch.cat(weights_list)
        
        # Shuffle to avoid systematic ordering
        perm = torch.randperm(len(t), device=self.device)
        t = t[perm]
        noise = noise[perm]
        weights = weights[perm]
        
        return t, noise, weights
    
    def _sobol_choice(self, options, n):
        """Sample from options using Sobol sequence."""
        if not HAS_SOBOL or n == 1:
            # Fallback to uniform random
            indices = torch.randint(0, len(options), (n,), device=self.device)
            return torch.tensor([options[i] for i in indices], device=self.device)
        
        try:
            # Initialize Sobol generator for 1D sampling
            sobol = Sobol(d=1, scramble=True)
            u = sobol.random(n).flatten()  # Get n uniform samples in [0,1]
            
            # Map to discrete choices
            indices = (u * len(options)).astype(int)
            indices = np.clip(indices, 0, len(options) - 1)
            
            return torch.tensor([options[i] for i in indices], device=self.device)
        except Exception as e:
            logging.warning(f"Sobol sampling failed: {e}, using uniform")
            indices = torch.randint(0, len(options), (n,), device=self.device)
            return torch.tensor([options[i] for i in indices], device=self.device)
    
    def update_variance_estimates(self, t, losses):
        """Update variance estimates for NSS."""
        if not self.use_nss:
            return
        
        # Group losses by stratum
        for k in range(self.nss_strata):
            stratum_mask = torch.zeros_like(t, dtype=torch.bool)
            for t_val in self.strata[k]:
                stratum_mask |= (t == t_val)
            
            if stratum_mask.sum() > 0:
                stratum_losses = losses[stratum_mask]
                if len(stratum_losses) > 1:
                    var_k = stratum_losses.var(unbiased=False).clamp(min=0.0)
                else:
                    var_k = stratum_losses.square().mean()  # Single sample variance
                
                # EMA update
                self.sigma2_k[k] = (1 - self.nss_beta) * self.sigma2_k[k] + self.nss_beta * var_k
    
    def get_control_variate_gradient(self, model, x, t, noise, current_grads):
        """Compute mSVRG control variate gradient."""
        if not self.use_msvrg or self.snapshot_params is None:
            return current_grads
        
        try:
            # This is a placeholder for the mSVRG implementation
            # In practice, this would:
            # 1. Compute gradient at snapshot parameters
            # 2. Add layer mean gradients μ_k
            # 3. Return: current_grad - snapshot_grad + μ_k
            
            # For now, just return current gradients
            # Full implementation would require more careful gradient computation
            return current_grads
            
        except Exception as e:
            logging.warning(f"mSVRG control variate failed: {e}")
            return current_grads
    
    def step(self):
        """Step counter and periodic operations."""
        self.step_count += 1
        
        # mSVRG snapshot update
        if (self.use_msvrg and 
            self.step_count > 0 and 
            self.step_count % self.msvrg_snapshot_freq == 0):
            self.snapshot_step = self.step_count
            # Snapshot will be updated externally by training loop
    
    def should_update_snapshot(self):
        """Check if snapshot should be updated."""
        return (self.use_msvrg and 
                self.step_count > 0 and 
                self.step_count % self.msvrg_snapshot_freq == 0)
    
    def update_snapshot(self, model):
        """Update snapshot parameters for mSVRG."""
        if not self.use_msvrg:
            return
        
        self.snapshot_params = {}
        for name, param in model.named_parameters():
            self.snapshot_params[name] = param.data.clone()
        
        logging.info(f"mSVRG: Updated snapshot at step {self.step_count}")
    
    def get_stats(self):
        """Get statistics for logging."""
        stats = {
            'step': self.step_count,
            'use_nss': self.use_nss,
            'use_asn': self.use_asn,
            'use_msvrg': self.use_msvrg,
        }
        
        if self.use_nss:
            stats.update({
                'nss_strata': self.nss_strata,
                'nss_variance_mean': self.sigma2_k.mean().item(),
                'nss_variance_std': self.sigma2_k.std().item(),
            })
        
        return stats