#!/usr/bin/env python
import cv2
import time
import random
import argparse
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision.utils as vutils

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        old_space = self.observation_space
        assert isinstance(old_space, spaces.Box)
        self.observation_space = spaces.Box(
            self.observation(old_space.low), self.observation(old_space.high),
            dtype=np.float32
        )

    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        # resize image
        new_obs = cv2.resize(
            observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (w, h, c) -> (c, w, h)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


def iterate_batches(envs: tt.List[gym.Env],
                    batch_size: int = BATCH_SIZE) -> tt.Generator[torch.Tensor, None, None]:
    batch = [e.reset()[0] for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        action = e.action_space.sample()
        obs, reward, terminated, truncated, _ = e.step(action)
        done = terminated or truncated
        
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            batch_np = np.array(batch, dtype=np.float32)
            # Normalising input to [-1..1] and convert to tensor
            yield torch.tensor(batch_np * 2.0 / 255.0 - 1.0)
            batch.clear()
        if done:
            e.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    parser.add_argument("--envs", nargs="+", default=["ALE/Breakout-v5", "ALE/AirRaid-v5", "ALE/Pong-v5"],
                       help="List of environment names")
    parser.add_argument("--max_iter", type=int, default=10000, help="Maximum iterations")
    args = parser.parse_args()

    device = torch.device(args.dev)
    log.info(f"Using device: {device}")
    
    # 检查可用的Atari环境
    log.info("Checking available Atari environments...")
    
    # 尝试创建环境
    envs = []
    for env_name in args.envs:
        try:
            env = gym.make(env_name)
            wrapped_env = InputWrapper(env)
            envs.append(wrapped_env)
            log.info(f"Successfully created environment: {env_name}")
        except gym.error.Error as e:
            log.warning(f"Failed to create {env_name}: {e}")
            log.info("Trying with render mode 'rgb_array'...")
            try:
                env = gym.make(env_name, render_mode='rgb_array')
                wrapped_env = InputWrapper(env)
                envs.append(wrapped_env)
                log.info(f"Successfully created with render mode: {env_name}")
            except Exception as e2:
                log.error(f"Completely failed to create {env_name}: {e2}")
    
    if len(envs) == 0:
        log.error("No environments created successfully!")
        log.info("\nAvailable Atari environments (try these instead):")
        # 列出可用的Atari环境
        from gymnasium import envs
        atari_envs = [spec.id for spec in envs.registry.values() 
                     if "ALE/" in spec.id or "Atari" in spec.id or "atari" in spec.id.lower()]
        for env_id in sorted(atari_envs)[:20]:  # 显示前20个
            log.info(f"  {env_id}")
        exit(1)
    
    shape = envs[0].observation_space.shape
    log.info(f"Input shape: {shape}")
    
    # 初始化网络
    net_discr = Discriminator(input_shape=shape).to(device)
    net_gener = Generator(output_shape=shape).to(device)
    
    # 输出模型信息
    d_params = sum(p.numel() for p in net_discr.parameters())
    g_params = sum(p.numel() for p in net_gener.parameters())
    log.info(f"Discriminator parameters: {d_params:,}")
    log.info(f"Generator parameters: {g_params:,}")
    
    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE,
                               betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE,
                               betas=(0.5, 0.999))
    writer = SummaryWriter()
    
    gen_losses = []
    dis_losses = []
    iter_no = 0
    
    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)
    ts_start = time.time()
    
    log.info("Starting training...")
    
    try:
        for batch_v in iterate_batches(envs):
            if iter_no >= args.max_iter:
                log.info(f"Reached maximum iterations: {args.max_iter}")
                break
                
            # fake samples, input is 4D: batch, filters, x, y
            gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
            gen_input_v.normal_(0, 1)
            gen_input_v = gen_input_v.to(device)
            batch_v = batch_v.to(device)
            gen_output_v = net_gener(gen_input_v)
            
            # train discriminator
            dis_optimizer.zero_grad()
            dis_output_true_v = net_discr(batch_v)
            dis_output_fake_v = net_discr(gen_output_v.detach())
            dis_loss = objective(dis_output_true_v, true_labels_v) + \
                       objective(dis_output_fake_v, fake_labels_v)
            dis_loss.backward()
            dis_optimizer.step()
            dis_losses.append(dis_loss.item())
            
            # train generator
            gen_optimizer.zero_grad()
            dis_output_v = net_discr(gen_output_v)
            gen_loss_v = objective(dis_output_v, true_labels_v)
            gen_loss_v.backward()
            gen_optimizer.step()
            gen_losses.append(gen_loss_v.item())
            
            iter_no += 1
            if iter_no % REPORT_EVERY_ITER == 0:
                dt = time.time() - ts_start
                log.info("Iter %d in %.2fs: gen_loss=%.3e, dis_loss=%.3e",
                         iter_no, dt, np.mean(gen_losses), np.mean(dis_losses))
                ts_start = time.time()
                writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
                writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
                gen_losses = []
                dis_losses = []
            if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
                img = vutils.make_grid(gen_output_v.data[:64], normalize=True)
                writer.add_image("fake", img, iter_no)
                img = vutils.make_grid(batch_v.data[:64], normalize=True)
                writer.add_image("real", img, iter_no)
                
                # 保存模型检查点
                torch.save({
                    'iteration': iter_no,
                    'generator_state_dict': net_gener.state_dict(),
                    'discriminator_state_dict': net_discr.state_dict(),
                    'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                    'dis_optimizer_state_dict': dis_optimizer.state_dict(),
                }, f"gan_checkpoint_{iter_no}.pth")
                log.info(f"Saved checkpoint at iteration {iter_no}")
    
    except KeyboardInterrupt:
        log.info("Training interrupted by user")
    except Exception as e:
        log.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        writer.close()
        for env in envs:
            env.close()
        log.info("Training completed")