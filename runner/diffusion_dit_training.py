import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from model.dit import DITModel
from model.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import torch.nn as nn
import torchvision.utils as tvu
from vrdit_controller import VRDiTController

def updateBN(model, s=1e-4):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s*torch.sign(m.weight.data))

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def get_rank():
    """Get rank safely, return 0 for non-distributed training."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class DITDiffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        
        if not hasattr(args, 'single_gpu') or not args.single_gpu:
            if not torch.distributed.is_initialized():
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                world_size = int(os.environ.get('WORLD_SIZE', 1))
                rank = int(os.environ.get('RANK', 0))
                
                torch.distributed.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank
                )
                
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f'cuda:{local_rank}')
            else:
                self.device = self.config.device
        else:
            self.device = self.config.device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        self.build_model()

    def build_model(self):
        args, config = self.args, self.config
        model = DITModel(config)

        if args.load_pruned_model is not None:
            print("Loading pruned model from {}".format(args.load_pruned_model))
            states = torch.load(args.load_pruned_model, map_location='cpu')

            if isinstance(states, torch.nn.Module):
                model = torch.load(args.load_pruned_model, map_location='cpu')
            elif isinstance(states, list):
                model = torch.load(args.base_pruned_model, map_location='cpu')
                weights_dict = {}
                for k, v in states[0].items():
                    new_k = k.replace('module.', '') if 'module' in k else k
                    weights_dict[new_k] = v
                model = model.to(self.device)
                model.load_state_dict(weights_dict, strict=True) 
                if args.use_ema and self.config.model.ema:
                    print("Loading EMA")
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                    self.ema_helper = ema_helper
                else:
                    self.ema_helper = None
            else:
                raise NotImplementedError
            self.model = model
        elif args.restore_from is not None and os.path.isfile(args.restore_from):
            ckpt = args.restore_from
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(
                ckpt,
                map_location='cpu',
            )
            if ".ckpt" in ckpt:
                model = model.to(self.device)
                model.load_state_dict(states, strict=True) 
            else:
                if isinstance(states[0], torch.nn.Module):
                    model = states[0]
                    model = model.to(self.device)
                else:
                    weights_dict = {}
                    for k, v in states[0].items():
                        new_k = k.replace('module.', '') if 'module' in k else k
                        weights_dict[new_k] = v
                    model = model.to(self.device)
                    model.load_state_dict(weights_dict, strict=True) 
                
                if args.use_ema and self.config.model.ema:
                    print("Loading EMA")
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                    self.ema_helper = ema_helper
                else:
                    self.ema_helper = None
        elif self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                ckpt = os.path.join(self.args.log_path, "ckpt.pth")
                states = torch.load(
                    ckpt,
                    map_location=self.config.device,
                )
            else:
                ckpt = os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    )
                states = torch.load(
                    ckpt,
                    map_location=self.config.device,
                )
            print("Loading checkpoint {}".format(ckpt))

            if isinstance(states[0], torch.nn.Module):
                model = states[0]
                model = model.to(self.device)
            else:
                weights_dict = {}
                for k, v in states[0].items():
                    new_k = k.replace('module.', '') if 'module' in k else k
                    weights_dict[new_k] = v
                model = model.to(self.device)
                model.load_state_dict(weights_dict, strict=True)

            if args.use_ema and self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
                self.ema_helper = ema_helper
            else:
                self.ema_helper = None
        elif self.args.train_from_scratch:
            return
        else:
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == "CELEBA":
                name = 'celeba'
            ckpt = get_ckpt_path(f"ema_{name}") 
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(ckpt, map_location=self.device)
            if isinstance(states, (list,tuple)):
                state_dict = states[0]
            else:
                state_dict = states
            
            # Handle key mismatch: add 'dit.' prefix if needed
            model_keys = set(model.state_dict().keys())
            state_keys = set(state_dict.keys())
            
            # Check if we need to add 'dit.' prefix
            if not any(key.startswith('dit.') for key in state_keys) and any(key.startswith('dit.') for key in model_keys):
                print("Adding 'dit.' prefix to state_dict keys...")
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = f"dit.{key}"
                    new_state_dict[new_key] = value
                state_dict = new_state_dict
            
            # Check if we need to remove 'dit.' prefix
            elif any(key.startswith('dit.') for key in state_keys) and not any(key.startswith('dit.') for key in model_keys):
                print("Removing 'dit.' prefix from state_dict keys...")
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('dit.'):
                        new_key = key[4:]  # Remove 'dit.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            # Try loading with strict=False first to see what keys are missing/unexpected
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                
                # If there are missing keys, try with strict=False
                if missing_keys or unexpected_keys:
                    print("Loading with strict=False due to key mismatches")
                    model.load_state_dict(state_dict, strict=False)
                else:
                    model.load_state_dict(state_dict, strict=True)
                    
            except Exception as e:
                print(f"Error loading state_dict: {e}")
                print("Attempting to load with strict=False...")
                model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
        self.model = model
    
    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        
        # Handle single GPU vs distributed training
        if hasattr(args, 'single_gpu') and args.single_gpu:
            train_loader = data.DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
                pin_memory=True
            )
        else:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(dataset)
            train_loader = data.DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.data.num_workers,
                pin_memory=True,
                sampler=train_sampler
            )
        if self.args.train_from_scratch:
            model = DITModel(config)
        else:
            model = self.model

        model = model.to(self.device)
        
        # Handle single GPU vs distributed training for model wrapping
        if not (hasattr(args, 'single_gpu') and args.single_gpu):
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[self.device], 
                output_device=self.device,
                find_unused_parameters=False  # VR-DiT uses standard DDP
            )

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
            
        # Initialize VR-DiT Controller
        self.vrdit_controller = VRDiTController(
            T=self.num_timesteps,
            device=self.device,
            # NSS parameters
            use_nss=getattr(config.training, 'use_nss', True),
            nss_strata=getattr(config.training, 'nss_strata', 4),
            nss_beta=getattr(config.training, 'nss_beta', 0.05),
            nss_use_weights=getattr(config.training, 'nss_use_weights', False),
            # ASN parameters
            use_asn=getattr(config.training, 'use_asn', True),
            asn_antithetic=getattr(config.training, 'asn_antithetic', True),
            asn_sobol=getattr(config.training, 'asn_sobol', True),
            # mSVRG parameters
            use_msvrg=getattr(config.training, 'use_msvrg', True),
            msvrg_snapshot_freq=getattr(config.training, 'msvrg_snapshot_freq', 500),
            msvrg_buffer_size=getattr(config.training, 'msvrg_buffer_size', 256),
        )
        
        # Enable VR-DiT if specified in config
        if getattr(config.training, 'use_vrdit', False):
            logging.info("VR-DiT (Variance-Reduced DiT) enabled")

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])
        states = [model.state_dict(),
                optimizer.state_dict(),]
                        
        if self.config.model.ema:
            states.append(ema_helper.state_dict())
        if self.args.train_from_scratch:
            torch.save(
                states,
                os.path.join(self.args.log_path, "ckpt_init.pth")
            )
        else:
            torch.save(
                model,
                os.path.join(self.args.log_path, "ckpt_init.pth")
            )
        
        if torch.distributed.is_initialized(): torch.distributed.barrier()
        model.eval()
        if get_rank() == 0: 
            os.makedirs(os.path.join(args.log_path, 'vis'), exist_ok=True)
            with torch.no_grad():
                n = 36
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)
                grid = tvu.make_grid(x)
                tvu.save_image(grid, os.path.join(args.log_path, 'vis', 'Init.png'))

        if torch.distributed.is_initialized(): torch.distributed.barrier()

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                self.vrdit_controller.step()

                x = x.to(self.device)
                y = y.to(self.device) if y is not None else None
                x = data_transform(self.config, x)
                b = self.betas

                # VR-DiT: Sample timesteps and noise using variance reduction
                t, e, weights = self.vrdit_controller.sample_timesteps_and_noise(n, x.shape)

                # VR-DiT uses full batch (no sample selection like AEGS)
                x_selected = x
                t_selected = t
                e_selected = e
                y_selected = y
                
                # Compute loss with importance sampling weights if using NSS
                loss_fn = loss_registry[config.model.type]
                
                if self.vrdit_controller.nss_use_weights and self.vrdit_controller.use_nss:
                    # Get per-sample losses for importance weighting
                    try:
                        per_sample_losses = loss_fn(model, x_selected, t_selected, e_selected, b, y_selected, keepdim=True)
                    except TypeError:
                        # Fallback for loss functions that don't support keepdim or y parameter
                        logging.warning("Loss function doesn't support keepdim/y parameter, using fallback")
                        per_sample_losses = loss_fn(model, x_selected, t_selected, e_selected, b, keepdim=True)
                    loss = (per_sample_losses * weights).mean()
                else:
                    try:
                        loss = loss_fn(model, x_selected, t_selected, e_selected, b, y_selected)
                    except TypeError:
                        # Fallback for loss functions that don't support y parameter
                        loss = loss_fn(model, x_selected, t_selected, e_selected, b)

                if get_rank() == 0: 
                    tb_logger.add_scalar("loss", loss, global_step=step)
                    
                    # Log VR-DiT statistics
                    vrdit_stats = self.vrdit_controller.get_stats()
                    for key, value in vrdit_stats.items():
                        tb_logger.add_scalar(f"vrdit/{key}", value, global_step=step)

                    logging.info(
                        f"step: {step}, loss: {loss.item():.6f}, "
                        f"batch_size: {n}, "
                        f"data_time: {data_time / (i+1):.4f}, "
                        f"vrdit_nss: {self.vrdit_controller.use_nss}, "
                        f"vrdit_asn: {self.vrdit_controller.use_asn}"
                    )

                optimizer.zero_grad()
                loss.backward()
                
                # VR-DiT: Update variance estimates for NSS
                if self.vrdit_controller.use_nss:
                    # Compute per-sample losses for variance estimation
                    with torch.no_grad():
                        if self.vrdit_controller.nss_use_weights:
                            # We already have per_sample_losses from above
                            self.vrdit_controller.update_variance_estimates(t_selected, per_sample_losses.detach())
                        else:
                            # Compute per-sample losses for variance tracking
                            try:
                                per_sample_losses = loss_fn(model, x_selected, t_selected, e_selected, b, y_selected, keepdim=True)
                            except TypeError:
                                # Fallback for loss functions that don't support keepdim or y parameter
                                per_sample_losses = loss_fn(model, x_selected, t_selected, e_selected, b, keepdim=True)
                            self.vrdit_controller.update_variance_estimates(t_selected, per_sample_losses.detach())
                
                # VR-DiT: Apply mSVRG control variates if enabled
                if self.vrdit_controller.use_msvrg:
                    try:
                        # Get current gradients
                        current_grads = []
                        for param in model.parameters():
                            if param.grad is not None:
                                current_grads.append(param.grad.data.clone())
                        
                        # Apply control variate (placeholder - full implementation needed)
                        # controlled_grads = self.vrdit_controller.get_control_variate_gradient(
                        #     model, x_selected, t_selected, e_selected, current_grads
                        # )
                        
                        # For now, just use current gradients
                        
                    except Exception as e:
                        logging.warning(f"VR-DiT mSVRG failed: {e}, using standard gradients")

                if self.args.sr:
                    updateBN(model)

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                
                optimizer.step()
                
                # VR-DiT: Update snapshot if needed
                if self.vrdit_controller.should_update_snapshot():
                    self.vrdit_controller.update_snapshot(model)

                if args.use_ema and self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    if get_rank() == 0: 
                        model.zero_grad()
                        
                        if hasattr(model, 'module'):
                            model_state_dict = model.module.state_dict()
                        else:
                            model_state_dict = model.state_dict()
                        
                        states = [
                            model_state_dict,
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if args.use_ema and self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        torch.save(states, os.path.join(args.log_path, f"ckpt_{step}.pth"))
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    

                data_start = time.time()
            if get_rank() == 0 and epoch % 100 == 0: 
                states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                            ]
                if args.use_ema and self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.args.log_path, "ckpt_ep{}.pth".format(epoch)),
                )


    def sample(self):
        self.model.eval()

        if self.args.fid:
            self.sample_fid(self.model)
        elif self.args.interpolation:
            self.sample_interpolation(self.model)
        elif self.args.sequence:
            self.sample_sequence(self.model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model):
        import torch
        torch.manual_seed(0)
        import random
        random.seed(0)
        import numpy as np
        np.random.seed(0)

        config = self.config
        
        os.makedirs(self.args.image_folder, exist_ok=True)
        
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 1000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                y = None
                if hasattr(config.model, 'num_classes') and config.model.num_classes > 0:
                    y = torch.randint(0, config.model.num_classes, (n,), device=self.device)

                x = self.sample_image(x, model, y)
                x = inverse_data_transform(config, x)

                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        y = None
        if hasattr(config.model, 'num_classes') and config.model.num_classes > 0:
            y = torch.randint(0, config.model.num_classes, (8,), device=self.device)

        with torch.no_grad():
            _, x = self.sample_image(x, model, y, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        y = None
        if hasattr(config.model, 'num_classes') and config.model.num_classes > 0:
            y = torch.randint(0, config.model.num_classes, (x.size(0),), device=self.device)

        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                batch_y = y[i:i+8] if y is not None else None
                xs.append(self.sample_image(x[i : i + 8], model, batch_y))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, y=None, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps_dit

            xs = generalized_steps_dit(x, seq, model, self.betas, eta=self.args.eta, y=y)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps_dit

            x = ddpm_steps_dit(x, seq, model, self.betas, y=y)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass

    def obtain_feature(self, x, step):
        b_size = x.size()[0]         
        x = x.to(self.device)
        x = data_transform(self.config, x)
        e = torch.randn_like(x)

        t = step * torch.ones(b_size).to(self.device)
        
        a = (1-self.betas).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        mid_feature, output = self.model(x, t.float(), return_mid=True)
        return x, mid_feature, output