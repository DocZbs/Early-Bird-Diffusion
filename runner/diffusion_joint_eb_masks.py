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
import torch.nn as nn
import torchvision.utils as tvu

def updateBN(model, s=1e-4):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(s*torch.sign(m.weight.data))  # L1

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
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
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


class Diffusion(object):
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
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        
        self.base_pruned_model = self.args.bird_pruned_model_path
        self.pruned_model_paths = f'checkpoints/logs/cifar10/model_0_ckpt_step_30000.pth' #todo: remove
        self.model = None
        self.ema_helper =None

        print("***************")
        print(f'sampling from timesteps {self.args.ts_lower_bound},{self.args.ts_upper_bound}')
        print("***************")
        
        self.build_model()
    
    def build_model(self):
        args, config = self.args, self.config
        # load the weight dicts for each pruned model, this will tell us if the item was pruned or not
        base_pruned_model_path = self.base_pruned_model
        sample_pruned_model_path = self.pruned_model_paths

        print("Loading pruned model from {}".format(base_pruned_model_path))

        states = torch.load(base_pruned_model_path, map_location='cpu')
        if args.sample:
            print(f'we are sampling! ')
            print(f' file path is {sample_pruned_model_path}')
            states = torch.load(sample_pruned_model_path, map_location='cpu')
        if isinstance(states, torch.nn.Module): # a simple pruned model 
            print('loading model states was nn.Module')
            # model = torch.load(base_pruned_model_paths, map_location='cpu')
            model =  torch.load(base_pruned_model_path, map_location='cpu')
        elif isinstance(states, list) and args.sample: # pruned model and training states
            print('states was list')
            model = torch.load(base_pruned_model_path,map_location='cpu')
            weights_dict = {}
            for k, v in states[0].items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v
            
            model = model.to(self.device)
            model.load_state_dict(weights_dict, strict=True) 
            # model = states[0]
            if args.use_ema and self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
                self.ema_helper = ema_helper
            else:
                self.ema_helper = None
        else:
            raise NotImplementedError

        # finalized model
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
        # if self.args.train_from_scratch:
        #     self.models = [Model(config) for model in self.models]
        
        self.model = self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        optimizer = get_optimizer(self.config,self.model.parameters()) 

        if self.config.model.ema:
            self.ema_helper = EMAHelper(mu=self.config.model.ema_rate) 
            self.ema_helper.register(self.model)
        else:
            self.ema_helper = None

        start_epoch, step = 0, 0

        if self.args.resume_training:
            resume_training_path = os.path.join(self.args.log_path,'model_ckpt_step_20000.pth')
            print(f'resuming training from: {resume_training_path}')
            states = torch.load(resume_training_path)
            self.model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                self.ema_helper.load_state_dict(states[4])
        states = [self.model.state_dict(),
                optimizer.state_dict(),]
                        
        if self.config.model.ema:
            states.append(self.ema_helper.state_dict())
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

        logging.info(
            f'sampling from timesteps {self.args.ts_lower_bound},{self.args.ts_upper_bound}'
        )

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                # TODO: the timesteps should be selected based on the high, medium, low characteristic
                # bird one: high noise
                # sample lower end
                t = torch.randint(
                    low=self.args.ts_lower_bound, high=self.args.ts_upper_bound+1, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.args.ts_upper_bound+ self.args.ts_lower_bound-t], dim=0)[:n]
                # Clamp timestep indices to the valid range [0, 999]
                t = torch.clamp(t, min=0, max=999)

                loss = loss_registry[config.model.type](self.model, x, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                self.model.zero_grad()
                loss.backward()

                if self.args.sr:
                    updateBN(self.model)
                
                try:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if args.use_ema and self.config.model.ema:
                    self.ema_helper.update(self.model)
                
                # save all birds
                if step % self.config.training.snapshot_freq == 0 or step == 1 or step %100000 == 0:
                    self.model.zero_grad()
                    states = [
                        self.model.state_dict() ,
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if args.use_ema and self.config.model.ema:
                        states.append(self.ema_helper.state_dict())
                    torch.save(
                        states,
                        os.path.join(self.args.log_path,
                                    f'model_ckpt_step_{step}.pth')
                    )


                data_start = time.time()


    def sample(self):

        # eval for all models
        for i in range(len(self.models)):
            print(f'setting model {i} eval')
            self.models[i].to(self.device)
            self.models[i].eval()

        if self.args.fid:
            self.sample_fid(self.models)
        # elif self.args.interpolation:
        #     self.sample_interpolation(self.model)
        # elif self.args.sequence:
        #     self.sample_sequence(self.model)
        # else:
        #     raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, models):
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

                x = self.sample_image(x,models)
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

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
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

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))
        

    def sample_image(self, x, models, last=True):
        # to do: make update
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
            from functions.denoising import generalized_steps_models

            xs = generalized_steps_models(x, seq, models, self.betas, eta=self.args.eta)
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


    def obtain_feature(self, x, step):
        b_size =x.size()[0]         
        x = x.to(self.device)
        x = data_transform(self.config, x)
        e = torch.randn_like(x)

        # antithetic sampling
        t = step*torch.ones(b_size).to(self.device)
            
        
        # t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        a = (1-self.betas).cumprod(dim=0).index_select(0, t.long()).view(-1, 1, 1, 1)
        x = x * a.sqrt() + e * (1.0 - a).sqrt()
        mid_feature, output = self.model(x, t.float(), return_mid=True)
        return x, mid_feature, output
        # if keepdim:
        #     return (e - output).square().sum(dim=(1, 2, 3))
        # else:
        #     return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
