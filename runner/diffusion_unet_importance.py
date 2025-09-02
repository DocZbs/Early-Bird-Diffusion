import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from model.diffusion import Model
from model.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from timestep_importance_sampler import TimestepImportanceSampler
import torchvision.utils as tvu

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

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


class UNetDiffusionImportance(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

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
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
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
        model = Model(config)
        
        if args.restore_from is not None and os.path.isfile(args.restore_from):
            ckpt = args.restore_from
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(ckpt, map_location=self.device)
            if isinstance(states[0], torch.nn.Module):
                model = states[0]
                model = model.to(self.device)
            else:
                model = model.to(self.device)
                model.load_state_dict(states[0], strict=True) 

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
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == "CELEBA":
                name = 'celeba'
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            states = torch.load(ckpt, map_location=self.device)
            if isinstance(states, (list, tuple)):
                model.load_state_dict(states[0])
            else:
                model.load_state_dict(states)
            model.to(self.device)
        
        self.model = model

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        model = self.model
        model = model.to(self.device)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
            
        # Initialize Timestep Importance Sampler
        if getattr(config.training, 'use_importance_sampling', False):
            self.importance_sampler = TimestepImportanceSampler(
                num_timesteps=self.num_timesteps,
                device=self.device,
                ema_decay=getattr(config.training, 'importance_ema_decay', 0.99),
                min_prob=getattr(config.training, 'importance_min_prob', 0.1),
                warmup_steps=getattr(config.training, 'importance_warmup_steps', 2000),
                update_freq=getattr(config.training, 'importance_update_freq', 200),
            )
            logging.info("Timestep Importance Sampling enabled")
        else:
            self.importance_sampler = None

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

        # Initial visualization
        model.eval()
        os.makedirs(os.path.join(args.log_path, 'vis'), exist_ok=True)
        with torch.no_grad():
            n = config.sampling.batch_size
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

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0    
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                
                if self.importance_sampler:
                    self.importance_sampler.step()

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # Sample timesteps using importance sampling if enabled
                if self.importance_sampler:
                    t, weights = self.importance_sampler.sample_timesteps(n)
                else:
                    # Standard uniform sampling
                    t = torch.randint(low=0, high=self.num_timesteps, size=(n,)).to(self.device)
                    weights = torch.ones(n, device=self.device)

                # Compute loss
                loss_fn = loss_registry[config.model.type]
                
                if self.importance_sampler and getattr(config.training, 'use_importance_weights', True):
                    # Get per-sample losses for importance weighting
                    try:
                        per_sample_losses = loss_fn(model, x, t, e, b, keepdim=True)
                    except TypeError:
                        # Fallback for loss functions that don't support keepdim
                        per_sample_losses = loss_fn(model, x, t, e, b)
                        if per_sample_losses.dim() == 0:
                            per_sample_losses = per_sample_losses.unsqueeze(0).repeat(n)
                    
                    # Apply importance weights
                    weighted_losses = per_sample_losses * weights
                    loss = weighted_losses.mean()
                    
                    # Update loss history for importance sampling
                    self.importance_sampler.update_loss_history(t, per_sample_losses.detach())
                else:
                    loss = loss_fn(model, x, t, e, b)
                    
                    # Still update loss history even if not using weights
                    if self.importance_sampler:
                        if loss.dim() == 0:
                            per_sample_losses = loss.unsqueeze(0).repeat(n)
                        else:
                            per_sample_losses = loss
                        self.importance_sampler.update_loss_history(t, per_sample_losses.detach())

                tb_logger.add_scalar("loss", loss, global_step=step)
                
                # Log importance sampling statistics
                if self.importance_sampler:
                    importance_stats = self.importance_sampler.get_stats()
                    for key, value in importance_stats.items():
                        tb_logger.add_scalar(f"importance/{key}", value, global_step=step)
     
                if step % 100 == 0:
                    log_msg = (
                        f"step: {step} (Ep={epoch}/{self.config.training.n_epochs}, "
                        f"Iter={i}/{len(train_loader)}), loss: {loss.item():.6f}, "
                        f"data time: {data_time / (i+1):.4f}"
                    )
                    
                    if self.importance_sampler:
                        log_msg += f", importance_sampling: {not importance_stats['warmup_phase']}"
                        if not importance_stats['warmup_phase']:
                            log_msg += f", entropy: {importance_stats['prob_entropy']:.3f}"
                    
                    logging.info(log_msg)

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)
                
                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    model.zero_grad()
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(states, os.path.join(self.args.log_path, f"ckpt_{step}.pth"))
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    
                data_start = time.time()

            # Visualization at end of epoch
            if epoch % 50 == 0:
                model.eval()
                with torch.no_grad():
                    n = config.sampling.batch_size
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
                    tvu.save_image(grid, os.path.join(args.log_path, 'vis', f'epoch-{epoch}.png'))

    def sample(self):
        model = self.model
        model.to(self.device)
        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedure not defined")

    def sample_fid(self, model):
        import torch
        torch.manual_seed(0)
        import random
        random.seed(0)
        import numpy as np
        np.random.seed(0)

        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
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

                x = self.sample_image(x, model)
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

        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

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

        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True):
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
            from functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
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
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self):
        pass