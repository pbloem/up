import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.decomposition import PCA

from former import TransformerBlock, SelfAttention, Attention

import up
from up.util import Reshape, kl_loss, vae_sample, gradient_norm, coords
from up import VAE, AE, ConditionalBlock, ConditionalTransformer, GTransformer

from former.util import d, here, tic, toc, sample_batch, enwik8_string, enwik8_bytes, estimate_compression
import former

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import torchvision
from torchvision import transforms as trf
from torchvision.utils import make_grid

import numpy as np

from argparse import ArgumentParser
import wandb

import random, sys, math, gzip, time, os

from tqdm import tqdm, trange

from collections import Counter

import fire
import copy

def weights_init(model : nn.Module, init_mult_max=1.0, mask_prob_max=0.0):

    if hasattr(model, 'alphas'):
        model.alphas = torch.bernoulli(torch.full_like(model.alphas, fill_value=0.5))
        model.betas  = torch.bernoulli(torch.full_like(model.betas, fill_value=0.5))

    for mod in model.modules():
        if type(mod) is nn.Linear or type(mod) is nn.Embedding or \
                type(mod) is nn.Conv2d or type(mod) is nn.ConvTranspose2d:

            init_mult = random.random() * init_mult_max
            mask_prob = random.random() * mask_prob_max

            # print(type(mod))
            # print(mod.weight.data[0])
            mod.reset_parameters()

            mod.weight.data *= init_mult
            # mod.weight.data **= mask_prob

            if mask_prob > 0.0:
                mask = torch.bernoulli(torch.full_like(mod.weight.data, fill_value=mask_prob)).to(torch.bool)
                mod.weight.data[mask] = 0.0

            # print(mod.weight.data[0])
            # print()


def sample_images(temperature=0.5, sample_batch_size=100, res=(64, 64),
         buffer_size=2000, latent_size=128, mlm_offset=0.0,
         reset_prob=0.01, num_batches=100_000, tags=[],
         debug=False, warmup=100_000, eval_every=5_000, print_every=50, gc=1.0,
         sequential=False, steps_per_sample=1, mlm_prob=0.15, ascii_only=False,
         init_mult_max=7.0, mask_prob_max=1.0, num_print=8,
       ):

    cmp_source = AE(ls=latent_size, mlm_offset=mlm_offset, res=res)

    if torch.cuda.is_available():
        cmp_source.cuda()

    buffer = torch.rand(size=(buffer_size, 3, *res), device=d())

    for i in (bar := trange(num_batches)):

        with torch.no_grad():

            # Re-initialize the parameters of source (i.e. sample a random source)
            weights_init(cmp_source, init_mult_max=init_mult_max, mask_prob_max=mask_prob_max)

            # slice a random selection of rows from the buffer (without replacement)
            iz = random.sample(range(buffer_size), sample_batch_size)
            z = buffer[iz, :]

            # replace some random rows with uniform random characters
            rows = torch.bernoulli(torch.full(size=(sample_batch_size, 1, 1, 1), fill_value=reset_prob))
            mask = rows.expand(sample_batch_size, 3, *res).to(torch.bool)

            rands = torch.rand(size=(sample_batch_size, 3, *res), device=d())

            # replace some random rows with uniform random character
            z[mask] = rands[mask]

            output = cmp_source(z)
            buffer[iz, :] = output

        if i % print_every == 0:

            grid = make_grid(buffer[:num_print**2], nrow=num_print).permute(1, 2, 0)
            plt.imshow(grid)
            plt.savefig(f'grid-{i:05}.png')

def sample_images2(res=(64, 64),
                  buffer_size=64, latent_size=16, mlm_offset=0.0,
                  init_mult_max=7.0, mask_prob_max=1.0, num_print=8,
                  ):

    cmp_source = AE(ls=latent_size, mlm_offset=mlm_offset, res=res)

    if torch.cuda.is_available():
        cmp_source.cuda()

    buffer = torch.rand(size=(buffer_size, 3, *res), device=d())

    for n in trange(30):

        grid = make_grid(buffer[:num_print ** 2], nrow=num_print).permute(1, 2, 0)
        plt.imshow(grid)
        plt.axis('off')
        plt.savefig(f'every-n-{n:03}.png')

        for i in range(buffer.size(0)):

            with torch.no_grad():

                # Re-initialize the parameters of source (i.e. sample a random source)
                weights_init(cmp_source, init_mult_max=init_mult_max, mask_prob_max=mask_prob_max)

                output = cmp_source(buffer[i:i+1])
                buffer[i] = output[0]

def plot_mnist(model, loader, numbatches=200, apply_pca=True):

    # gather up first 200 batches into one big tensor
    images, labels = [], []
    for i, (ims, lbs) in enumerate(loader):
        images.append(ims)
        labels.append(lbs)

        if i > numbatches:
            break

    images, labels = torch.cat(images, dim=0), torch.cat(labels, dim=0)

    if torch.cuda.is_available():
        images = images.cuda()

    n, c, h, w = images.size()
    assert images.size(1) == 1

    z = model.encoder(images.expand(n, 3, h, w))
    n, c = z.size()
    z = z[:, :c//2] # only the means
    latents = z.detach().cpu().numpy()

    if apply_pca:
        pca = PCA(n_components=2)
        pca.fit(latents)
        latents = pca.transform(latents)
    else:
        latents = latents[:, :2]

    xrange = latents[:, 0].min().item(), latents[:, 0].max().item()
    yrange = latents[:, 1].min().item(), latents[:, 1].max().item()
    size = 1.0 * max(xrange[1] - xrange[0], yrange[1] - yrange[0]) / math.sqrt(n)
    # Change 0.75 to any value between ~ 0.5 and 1.5 to make the digits smaller or bigger

    fig = plt.figure(figsize=(16, 16))

    # colormap for the images
    norm = mpl.colors.Normalize(vmin=0, vmax=9)
    cmap = mpl.cm.get_cmap('tab10')

    for i in range(n):
        x, y = latents[i, 0:2]
        l = labels[i]

        im = images[i, :].cpu()
        alpha_im = im.permute(1, 2, 0).numpy()
        color = cmap(norm(l))
        color_im = np.asarray(color)[None, None, :3]
        color_im = np.broadcast_to(color_im, (h, w, 3))
        # -- To make the digits transparent we make them solid color images and use the
        #    actual data as an alpha channel.
        #    color_im: 3-channel color image, with solid color corresponding to class
        #    alpha_im: 1-channel grayscale image corrsponding to input data

        im = np.concatenate([color_im, alpha_im], axis=2)
        plt.imshow(im, extent=(x, x + size, y, y + size))

    plt.xlim(xrange[0], xrange[1] + size)
    plt.ylim(yrange[0], yrange[1] + size)
    print('x range ', *xrange)
    print('y range ', *yrange)

    plt.axis('off')


def vae(res=(32, 32), cmp_latent_size=128, latent_size=16, mlm_offset=0.0, init_mult_max=7.0, mask_prob_max=1.0,
        num_print=8, num_batches=1_000_000, reset_prob=0.01, buffer_size=64, print_every=50, lr=1e-4, debug=False):

    wd = wandb.init(
        project='prior-vae',
        config=locals(),
        mode= 'disabled' if debug else 'online'
    )

    cmp_source = AE(ls=cmp_latent_size, mlm_offset=mlm_offset, res=res)
    model = MnistVAE(latent_size=latent_size)

    if torch.cuda.is_available():
        cmp_source.cuda()
        model.cuda()

    # Load MNIST for zero-shot plotting
    transforms = trf.Compose([
        trf.Resize(size=res),
        trf.ToTensor()
    ])

    test = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True, num_workers=0)

    buffer = torch.rand(size=(buffer_size, 3, *res), device=d())

    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    for n in (bar := trange(num_batches)):

        if n % print_every == 0:

            plt.figure()

            # print the buffer
            grid = make_grid(buffer[:num_print ** 2], nrow=num_print).permute(1, 2, 0)
            plt.imshow(grid.cpu().numpy())
            plt.axis('off')
            plt.savefig(f'vae-buffer-{n:06}.png')

            # print a sample from the model
            with torch.no_grad():
                samples = model.sample(n=num_print**2).detach()
                samples = make_grid(samples, nrow=num_print).permute(1, 2, 0)
            plt.imshow(samples.cpu().numpy())
            plt.axis('off')
            plt.savefig(f'vae-samples-{n:06}.png')

            plt.figure()
            # print the MNIST data in the latent space
            plot_mnist(model, testloader)
            plt.savefig(f'vae-mnist-{n:06}.png')

        # Resample the buffer
        # -- This is done one instance at a time for the moment, which is a little inefficient.
        for i in range(buffer.size(0)):

            with torch.no_grad():

                # Re-initialize the parameters of source (i.e. sample a random source)
                weights_init(cmp_source, init_mult_max=init_mult_max, mask_prob_max=mask_prob_max)

                output = cmp_source(buffer[i:i+1])
                buffer[i] = output[0]

        # Train on the buffer
        o, kl = model(buffer)

        rec = F.binary_cross_entropy(o, buffer, reduction='none')
        rec = rec.view(buffer_size, -1).sum(dim=1)

        loss = (rec + kl).mean()  # sum the losses and take the mean
        loss.backward()

        opt.step()
        opt.zero_grad()

        bar.set_postfix({'loss':loss.item()})
        wandb.log({'loss': loss.item()})

if __name__ == '__main__':
    fire.Fire()