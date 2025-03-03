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

import threading
import queue


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

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
        
        # should be biggest then medium then smallest
        DATASET = "cifar10"
        self.base_pruned_model_paths= [
            # distance <= 0.2
            # f'checkpoints/pruned/overlap_method_NEW/{DATASET}_0_260/models/{DATASET}-0.3-magnitude-5.pth',
            # f'checkpoints/pruned/overlap_method_NEW/{DATASET}_240_460/models/{DATASET}-0.6-magnitude-6.pth',
            # f'checkpoints/pruned/overlap_method_NEW/{DATASET}_440_1000/models/{DATASET}-0.8-magnitude-4.pth',

            # distance <= 0.25
            f'checkpoints/pruned/overlap_method_NEW/{DATASET}_0_260/models/{DATASET}-0.3-magnitude-4.pth',
            f'checkpoints/pruned/overlap_method_NEW/{DATASET}_240_460/models/{DATASET}-0.6-magnitude-5.pth',
            f'checkpoints/pruned/overlap_method_NEW/{DATASET}_440_1000/models/{DATASET}-0.8-magnitude-4.pth',

            # earliest
            # f'checkpoints/pruned/overlap_method_NEW/{DATASET}_0_260/models/{DATASET}-0.3-magnitude-0.pth',
            # f'checkpoints/pruned/overlap_method_NEW/{DATASET}_240_460/models/{DATASET}-0.6-magnitude-0.pth',
            # f'checkpoints/pruned/overlap_method_NEW/{DATASET}_440_1000/models/{DATASET}-0.8-magnitude-0.pth',
            
            # non eb no overlap & eb reduced window
            # f'checkpoints/pruned/overlap_method/{DATASET}_eb_03/models/{DATASET}_eb_03-0.3-magnitude-0.pth',
            # f'checkpoints/pruned/overlap_method/{DATASET}_eb_06/models/{DATASET}_eb_06-0.6-magnitude-0.pth',
            # f'checkpoints/pruned/overlap_method/{DATASET}_eb_08/models/{DATASET}_eb_08-0.8-magnitude-0.pth',

            # eb no overlap
            # f'checkpoints/pruned/{DATASET}_eb_03/models/{DATASET}_eb_03-0.3-magnitude-9.pth',
            # f'checkpoints/pruned/{DATASET}_eb_06/models/{DATASET}_eb_06-0.6-magnitude-11.pth',
            # f'checkpoints/pruned/overlap_method/{DATASET}_eb_08/models/{DATASET}_eb_08-0.8-magnitude-7.pth',

            # f'checkpoints/pruned/{DATASET}_eb_03/models/{DATASET}_eb_03-0.3-magnitude-0.pth',
            # f'checkpoints/pruned/{DATASET}_eb_06/models/{DATASET}_eb_06-0.6-magnitude-0.pth',
            # f'checkpoints/pruned/overlap_method/{DATASET}_eb_08/models/{DATASET}_eb_08-0.8-magnitude-0.pth',
            
            # non eb random prune
            # f'checkpoints/pruned/overlap_method/{DATASET}_non_eb_03/models/{DATASET}_eb_03-0.3-random-0.pth',
            # f'checkpoints/pruned/overlap_method/{DATASET}_non_eb_06/models/{DATASET}_eb_06-0.6-random-0.pth',
            # f'checkpoints/pruned/overlap_method/{DATASET}_non_eb_08/models/{DATASET}_eb_08-0.8-random-0.pth',

            # reduced overlap
            # f'checkpoints/pruned/overlap_method/{DATASET}_non_eb_03/models/{DATASET}_eb_03-0.3-random-0.pth',
            # f'checkpoints/pruned/overlap_method/{DATASET}_non_eb_06/models/{DATASET}_eb_06-0.6-random-0.pth',
            # f'checkpoints/pruned/overlap_method/{DATASET}_non_eb_08/models/{DATASET}_eb_08-0.8-random-0.pth',

            # baseline
            # f'checkpoints/pruned/baseline/cifar10-0.3.pth',
            # f'checkpoints/pruned/baseline/cifar10-0.6.pth',
            # f'checkpoints/pruned/baseline/cifar10-0.8.pth',
        ]
        self.pruned_model_paths = [
            # f'checkpoints/logs/cifar10_30_ts0-240_no_overlap/model_ckpt_step_{200000}.pth', 
            # f'checkpoints/logs/cifar10_60_ts240-440_no_overlap/model_ckpt_step_{200000}.pth',
            # f'checkpoints/logs/cifar10_80_ts440-1000_no_overlap/model_ckpt_step_{200000}.pth'

            # f'checkpoints/logs/redo_bird_30/model_ckpt_step_{200000}.pth', 
            # f'checkpoints/logs/cifar10_bird_60_ts240-460/model_ckpt_step_{200000}.pth',
            # f'checkpoints/logs/cifar10_bird_80_ts440-1000/model_ckpt_step_{200000}.pth'

            # earlybird
            # f'checkpoints/logs/eb_30_0_260/model_ckpt_step_{200000}.pth',
            # f'checkpoints/logs/eb_60_240_460/model_ckpt_step_{250000}.pth',
            # f'checkpoints/logs/eb_80_440_1000/model_ckpt_step_{300000}.pth'

            # f'checkpoints/logs/eb_30_dist_025/model_ckpt_step_{200000}.pth',
            # f'checkpoints/logs/eb_60_dist_025/model_ckpt_step_{200000}.pth',
            # f'checkpoints/logs/eb_80_440_1000/model_ckpt_step_{200000}.pth'

            # eb
            # f'checkpoints/logs/eb_30_0_260/model_ckpt_step_{200000}.pth',
            # f'checkpoints/logs/eb_60_240_460/model_ckpt_step_{200000}.pth',
            # f'checkpoints/logs/eb_80_440_1000/model_ckpt_step_{200000}.pth',

            # earliest bird
            f'checkpoints/logs/earliest_bird_30_0_260/model_ckpt_step_{200000}.pth',
            f'checkpoints/logs/earliest_bird_60_240_460/model_ckpt_step_{250000}.pth',
            f'checkpoints/logs/earliest_bird_80_240_460/model_ckpt_step_{350000}.pth',


            # window reduction
            # f'checkpoints/logs/cifar10_bird_30_ts0-240/model_ckpt_step_{200000}.pth', 
            # f'checkpoints/logs/cifar10_bird_60_ts240-440/model_ckpt_step_{200000}.pth',
            # f'checkpoints/logs/cifar10_bird_80_ts440-1000/model_ckpt_step_{200000}.pth'

            # non eb random
            # f'checkpoints/logs/cifar10_NON_bird_30_ts0-260/model_ckpt_step_{200000}.pth', 
            # f'checkpoints/logs/cifar10_NON_bird_60_ts240-460/model_ckpt_step_{200000}.pth',
            # f'checkpoints/logs/cifar10_NON_bird_80_ts440-1000/model_ckpt_step_{200000}.pth'

            # reduced overlap
            # f'checkpoints/logs/cifar10_bird_30_ts0-160_reduced_overlap/model_ckpt_step_{200000}.pth', 
            # f'checkpoints/logs/cifar10_bird_60_ts140-360_reduced_window/model_ckpt_step_{200000}.pth', 
            # f'checkpoints/logs/cifar10_bird_80_ts340-1000_reduced_window/model_ckpt_step_{200000}.pth', 

            # increased overlap
            # f'checkpoints/logs/cifar10_bird_30_ts0-360_increased_window/model_ckpt_step_{200000}.pth', 
            # f'checkpoints/logs/cifar10_bird_60_ts340-660_increased_window/model_ckpt_step_{200000}.pth', 
            # f'checkpoints/logs/cifar10_bird_80_ts540-1000_increased_window/model_ckpt_step_{200000}.pth', 
        ]
        self.models = []
        self.ema_helpers = []
        
        self.build_model()
    
    def build_model(self):
        args, config = self.args, self.config
        models = []
        print(f'models: {models}')

        # load the weight dicts for each pruned model, this will tell us if the item was pruned or not
        for i in range(len(self.base_pruned_model_paths)):
            base_pruned_model_path = self.base_pruned_model_paths[i]
            sample_pruned_model_path = self.pruned_model_paths[i]
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
                models.append(model)
                print(f' len models: {len(models)}')
            elif isinstance(states, list) and args.sample: # pruned model and training states
                print('states was list')
                model = torch.load(base_pruned_model_path,map_location='cpu')
                weights_dict = {}
                for k, v in states[0].items():
                    new_k = k.replace('module.', '') if 'module' in k else k
                    weights_dict[new_k] = v
                
                model = model.to(self.device)
                model.load_state_dict(weights_dict, strict=True) 
                models.append(model)
                # model = states[0]
                if args.use_ema and self.config.model.ema:
                    print("HERE")
                    print("Loading EMA")
                    ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                    ema_helper.register(model)
                    ema_helper.load_state_dict(states[-1])
                    ema_helper.ema(model)
                    self.ema_helpers.append(ema_helper)
                else:
                    self.ema_helper = None
            else:
                raise NotImplementedError
        print(f'models: {len(models)}')
        self.models = models
    
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
        
        for model in self.models:
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)

        optimizers = [
            get_optimizer(self.config,model.parameters()) for model in self.models
        ]

        if self.config.model.ema:
            for model in self.models:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate) 
                ema_helper.load_state_dict(model.state_dict())
                ema_helper.register(model)
                ema_helper.ema(model)
                self.ema_helpers.append(ema_helper)
        else:
            self.ema_helpers = None

        start_epoch, step = 0, 0

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                n = x.size(0)
                data_time += time.time() - data_start
                for model in self.models:
                    model.train()
                step += 1

                x = x.to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                # TODO: the timesteps should be selected based on the high, medium, low characteristic
                for mask_index in range(len(self.models)):
                    if mask_index == 0:
                        # sample lower end
                        t = torch.randint(
                            low=0, high=300, size=(n // 2 + 1,)
                        ).to(self.device)
                        t = torch.cat([t, 300- t - 1], dim=0)[:n]
                    elif mask_index == 1:
                        # sample middle end
                        # sample lower end
                        t = torch.randint(
                            low=300, high=500, size=(n // 2 + 1,)
                        ).to(self.device)
                        t = torch.cat([t, 600- t - 1], dim=0)[:n]
                    else:
                        # sample higher end
                        # sample lower end
                        t = torch.randint(
                            low=500, high=1000, size=(n // 2 + 1,)
                        ).to(self.device)
                        t = torch.cat([t, 1500- t - 1], dim=0)[:n]

                    loss = loss_registry[config.model.type](self.models[mask_index], x, t, e, b)

                    tb_logger.add_scalar("loss", loss, global_step=step)

                    logging.info(
                        f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                    )

                    self.models[mask_index].zero_grad()
                    loss.backward()

                    if self.args.sr:
                        updateBN(self.models[mask_index])
                    
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            self.models[mask_index].parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizers[mask_index].step()

                    if args.use_ema and self.config.model.ema:
                        self.ema_helpers[mask_index].update(self.models[mask_index])
                    
                    # save all birds
                    if step % self.config.training.snapshot_freq == 0 or step == 1 or step %10000 == 0:
                        for i in range(len(self.models)):
                            self.models[i].zero_grad()
                            states = [
                                self.models[i].state_dict() ,
                                optimizers[i].state_dict(),
                                epoch,
                                step,
                            ]
                            if args.use_ema and self.config.model.ema:
                                ema_helper_i = self.ema_helpers[i]
                                states.append(ema_helper_i.state_dict())
                            torch.save(
                                states,
                                os.path.join(self.args.log_path,
                                            f'model_{i}_ckpt_step_{step}.pth')
                            )


                data_start = time.time()


    def sample(self):
        # model = Model(self.config)

        # if not self.args.use_pretrained:
        #     if getattr(self.config.sampling, "ckpt_id", None) is None:
        #         states = torch.load(
        #             os.path.join(self.args.log_path, "ckpt.pth"),
        #             map_location=self.config.device,
        #         )
        #     else:
        #         states = torch.load(
        #             os.path.join(
        #                 self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
        #             ),
        #             map_location=self.config.device,
        #         )
        #     model = model.to(self.device)
        #     model = torch.nn.DataParallel(model)
        #     # weights_dict = {}
        #     # for k, v in states[0].items():
        #     #     new_k = k.replace('module.', '') if 'module' in k else k
        #     #     weights_dict[new_k] = v
        #     model.load_state_dict(states[0], strict=True)

        #     if self.config.model.ema:
        #         ema_helper = EMAHelper(mu=self.config.model.ema_rate)
        #         ema_helper.register(model)
        #         ema_helper.load_state_dict(states[-1])
        #         ema_helper.ema(model)
        #     else:
        #         ema_helper = None
        # else:
        #     # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        #     if self.config.data.dataset == "CIFAR10":
        #         name = "cifar10"
        #     elif self.config.data.dataset == "LSUN":
        #         name = f"lsun_{self.config.data.category}"
        #     else:
        #         raise ValueError
        #     ckpt = get_ckpt_path(f"ema_{name}")
        #     print("Loading checkpoint {}".format(ckpt))
        #     model.load_state_dict(torch.load(ckpt, map_location=self.device))
        #     model.to(self.device)
        #     model = torch.nn.DataParallel(model)

        # eval for all models
        for i in range(len(self.models)):
            print(f'setting model {i} eval')
            self.models[i].to(self.device)
            self.models[i].eval()

        if self.args.fid:
            self.sample_fid(self.models)
            # self.pipelined_sampling(self.models)
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
        print(f'img folder: {self.args.image_folder}')
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

    
    def pipelined_sampling(self, models):
        print(f'sampling batch size: {self.config.sampling.batch_size}')
        print('trying pipelined sampling')

        # Seed for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)

        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        print('starting', flush=True)

        n = config.sampling.batch_size
        num_batches = n_rounds

        # Initialize queues between models
        queue_small_medium = queue.Queue(maxsize=5)
        queue_medium_large = queue.Queue(maxsize=5)
        queue_large_final = queue.Queue()

        device = self.device  
        betas = self.betas

        # define the timestep regions
        T = self.num_timesteps - 1
        t_small_start = T
        t_small_end = 440
        t_medium_start = 439
        t_medium_end = 240
        t_large_start = 239
        t_large_end = -1  # timesteps go down to zero

        # init the final queue to collect results from the large model
        queue_final_result = queue.Queue()

        # Start worker threads for each model
        small_model_thread = threading.Thread(
            target=shared_model_worker,
            args=(models[2], queue_small_medium, queue_medium_large, device, 'Small Model', betas, t_small_start, t_small_end),
            kwargs={'eta': self.args.eta}
        )
        medium_model_thread = threading.Thread(
            target=shared_model_worker,
            args=(models[1], queue_medium_large, queue_large_final, device, 'Medium Model', betas, t_medium_start, t_medium_end),
            kwargs={'eta': self.args.eta}
        )
        large_model_thread = threading.Thread(
            target=shared_model_worker,
            args=(models[0], queue_large_final, queue_final_result, device, 'Large Model', betas, t_large_start, t_large_end),
            kwargs={'eta': self.args.eta}
        )

        small_model_thread.start()
        medium_model_thread.start()
        large_model_thread.start()

        batch_ids = []

        start_time = time.time()

        # enqueue batches into the pipeline
        for batch_id in range(num_batches):
            x = torch.randn(
                n,
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device='cpu',  # Start with CPU, models will move data to GPU as needed
            )
            # Put data into the small model's input queue
            queue_small_medium.put((batch_id, x))
            print(f"Main thread enqueued batch {batch_id}", flush=True)
            batch_ids.append(batch_id)

        # wait for all batches to be processed
        queue_small_medium.join()
        queue_medium_large.join()
        queue_large_final.join()

        # send stop signals to workers
        queue_small_medium.put(None)
        queue_medium_large.put(None)
        queue_large_final.put(None)

        # wait for worker threads to finish
        small_model_thread.join()
        medium_model_thread.join()
        large_model_thread.join()

        # collect final results from the queue_final_result
        final_results = {}
        for _ in batch_ids:
            batch_id, x = queue_final_result.get()
            final_results[batch_id] = x
            queue_final_result.task_done()

        # save the final images
        for batch_id in sorted(final_results.keys()):
            x = final_results[batch_id]
            x = inverse_data_transform(config, x)
            for i in range(n):
                tvu.save_image(
                    x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                )
                img_id += 1
        end_time = time.time()
        execution_time = end_time-start_time
        print(f'total time: {execution_time}')

        print("Pipelined sampling completed.", flush=True)

def shared_model_worker(model, input_queue, output_queue, device, model_name, betas, t_start, t_end, **kwargs):
    """Worker function for each model."""
    try:
        logging.info(f"{model_name} worker started")

        while True:
            data = input_queue.get()
            if data is None:  # Stop signal
                logging.info(f"{model_name} worker stopping")
                input_queue.task_done()
                break

            batch_id, x = data  # Unpack the data
            logging.info(f"{model_name} processing batch {batch_id}")

            with torch.no_grad():
                x = x.to(device)
                n = x.size(0)

                # Generate the sequence of timesteps for this model
                seq = list(range(t_start, t_end, -1)) if t_start > t_end else list(range(t_start, t_end))

                for idx, t in enumerate(seq):
                    t_tensor = torch.full((n,), t, device=device, dtype=torch.int64)
                    next_t = seq[idx + 1] if idx + 1 < len(seq) else t_end
                    next_t_tensor = torch.full((n,), next_t, device=device, dtype=torch.int64)

                    at = compute_alpha(betas, t_tensor)
                    at_next = compute_alpha(betas, next_t_tensor)
                    et = model(x, t_tensor)
                    x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()

                    c1 = kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    x_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                    x = x_next  # Update x for the next timestep

                x = x.cpu()  # Move to CPU after processing assigned timesteps

            # Pass the result to the next stage
            output_queue.put((batch_id, x))
            input_queue.task_done()

    except Exception as e:
        logging.error(f"Error in {model_name} worker: {str(e)}")