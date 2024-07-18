import torch
from torch import nn
import torch.nn.functional as F

from former import TransformerBlock, SelfAttention, Attention
from former.util import d

import random, math

from .util import Reshape, kl_loss, vae_sample, coords, Lambda


class TransformerBlock(nn.Module):
    """
    A straightforward transformer block.
    """

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.1,
                 pos_embedding=None, sa_kwargs={}, nl=torch.relu):
        super().__init__()

        self.nl = Lambda(lambda x : nl(x))

        self.attention = SelfAttention(emb, heads=heads, mask=mask, **sa_kwargs)

        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            self.nl,
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x


class VAE(nn.Module):
    """
    VAE designed specifically for the MNIST resolution (28, 28), though in color.
    """

    def __init__(self, a=16, b=32, c=128, latent_size=2, res=(32, 32), h=256):
        super().__init__()

        self.latent_size = latent_size

        self.res = res
        self.mr = mr = res[0] // 2**3, res[1] // 2**3

        self.encoder = nn.Sequential(
            nn.Conv2d(3, a, (3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(a, b, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(b, b, (3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(b, c, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c, c, (3, 3), padding=1), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(c * mr[0] * mr[1], h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 2 * latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, c * mr[0] * mr[1]), nn.ReLU(),
            Reshape((c, mr[0], mr[1])),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(c, b, (3, 3), padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(b, a, (3, 3), padding=1), nn.ReLU(),  # note the padding
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ConvTranspose2d(a, 3, (3, 3), padding=1), nn.Sigmoid()
        )

    def forward(self, images):
        b, c, h, w = images.size()

        # forward pass
        z = self.encoder(images)

        # - split z into mean and sigma
        _, c = z.size()
        zmean, zsig = z[:, :c//2], z[:, c//2:]
        kl = kl_loss(zmean, zsig)

        zsample = vae_sample(zmean, zsig)

        return self.decoder(zsample), kl

    def sample(self, n):

        zs = torch.randn((n, self.latent_size), device=d())

        return self.decoder(zs)

class AE(nn.Module):

    def __init__(self, a=16, b=32, c=128, ls=2, krnl=3, res=(64, 64), mlm_offset=0.0, ln_params=True, num_mids=3):
        super().__init__()

        self.latent_size = ls
        self.mlm_offset = mlm_offset

        self.res = res
        self.mr = mr = res[0] // 2**3, res[1] // 2**3

        # non-linearity
        self.nl = torch.sign
        # self.nl = lambda x : torch.sigmoid(x * 1e3) # torch.sign, F.relu
        # -- A sigmoid nonlinearity with a temperature parameter offers a nice way to tune between high and low frequency
        #    structures

        pm = 'zeros'

        pad = krnl//2
        krnl = (krnl, krnl)

        self.coords0 = coords(res[0],      res[1])
        self.coords1 = coords(res[0] // 2, res[1] // 2)
        self.coords2 = coords(res[0] // 4, res[1] // 4)
        self.coords3 = coords(res[0] // 8, res[1] // 8)

        self.conve11 = nn.Conv2d(3+2, a, krnl, padding=pad, padding_mode=pm)
        self.conve1point = nn.Conv2d(a+2, a, (1, 1), padding=0)

        self.ln1 = nn.LayerNorm(a, elementwise_affine=ln_params)

        self.conve21 = nn.Conv2d(a + 2, b, krnl, padding=pad, padding_mode=pm)
        self.conve22 = nn.Conv2d(b+2, b, krnl, padding=pad, padding_mode=pm)
        self.conve2point = nn.Conv2d(b+2, b, (1, 1), padding=0)

        self.ln2 = nn.LayerNorm(b, elementwise_affine=ln_params)

        self.conve31 = nn.Conv2d(b + 2, c, krnl, padding=pad, padding_mode=pm)
        self.conve32 = nn.Conv2d(c+2, c, krnl, padding=pad, padding_mode=pm)
        self.conve3point = nn.Conv2d(c+2, c, (1, 1), padding=0)

        self.ln3 = nn.LayerNorm(c, elementwise_affine=ln_params)

        self.line = nn.Linear(c * mr[0] * mr[1], ls)

        mids = [nn.Linear(ls, ls) for _ in range(num_mids)]
        self.mids = nn.ModuleList(mids)

        self.lind = nn.Linear(ls, c * mr[0] * mr[1])

        self.convd3point = nn.Conv2d(c + 2, c, (1, 1), padding=0)

        self.ln4 = nn.LayerNorm(c, elementwise_affine=ln_params)

        self.convd32 = nn.ConvTranspose2d(c + 2, c, krnl, padding=pad)
        self.convd31 = nn.ConvTranspose2d(c+2, b, krnl, padding=pad)

        self.convd2point = nn.Conv2d(b + 2, b, (1, 1), padding=0)

        self.ln5 = nn.LayerNorm(b, elementwise_affine=ln_params)

        self.convd22 =nn.ConvTranspose2d(b + 2, b, krnl, padding=pad)
        self.convd21 =nn.ConvTranspose2d(b +2, a, krnl, padding=pad)

        self.convd1point = nn.Conv2d(a + 2, a, (1, 1), padding=0)

        self.ln6 = nn.LayerNorm(a, elementwise_affine=ln_params)

        self.convd11 =nn.ConvTranspose2d(a + 2, 4, krnl, padding=pad)

        self.alphas = torch.tensor([1., 1., 1.])
        self.betas  = torch.tensor([1., 1., 1.])

        # -- We have four output channels. The fourth is a probability distribution that tells us how the input and
        #    and output are mixed in the sample.

    def coordconv(self, img, coords):
        b, c, h, w = img.size()
        assert coords.size() == (1, 2, h, w)
        return torch.cat([img, coords.expand(b, 2, h, w)], dim=1)

    def forward(self, img):

        b, c, h, w = img.size()

        # forward pass
        x = img

        x = self.coordconv(img, self.coords0)
        x = self.nl(self.conve11(x))

        x = self.coordconv(x, self.coords0)
        x = x_e11 = F.relu(self.conve1point(x))

        x = F.max_pool2d(x, kernel_size=2)
        x = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.coordconv(x, self.coords1)
        x = self.nl(self.conve21(x))

        x = self.coordconv(x, self.coords1)
        x = self.nl(self.conve22(x))

        x = self.coordconv(x, self.coords1)
        x = x_e22 = self.nl(self.conve2point(x))

        x = F.max_pool2d(x, kernel_size=2)
        x = self.nl(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.coordconv(x, self.coords2)
        x = self.nl(self.conve31(x))

        x = self.coordconv(x, self.coords2)
        x = self.nl(self.conve32(x))

        x = self.coordconv(x, self.coords2)
        x = x_e32 = self.nl(self.conve3point(x))

        x = F.max_pool2d(x, kernel_size=2)
        x = self.ln3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.line(x.reshape(b, -1))

        # Middle layers of the unet
        for mid in self.mids:
            x = self.nl(mid(x))

        x = self.lind(x).reshape(b, -1, *self.mr)

        x = F.upsample_bilinear(x, scale_factor=2) # --
        x = self.ln4(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.coordconv(x, self.coords2)    # point
        x = self.nl(self.convd3point(x))

        x = x * self.betas[0] + x_e32 * self.alphas[0] # residual

        x = self.coordconv(x, self.coords2)
        x = self.nl(self.convd32(x))

        x = self.coordconv(x, self.coords2)
        x = self.nl(self.convd31(x))

        x = F.upsample_bilinear(x, scale_factor=2) # --
        x = self.ln5(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.coordconv(x, self.coords1)    # point
        x = self.nl(self.convd2point(x))

        x = x * self.betas[1] + x_e22 * self.alphas[1] # res

        x = self.coordconv(x, self.coords1)
        x = self.nl(self.convd22(x))

        x = self.coordconv(x, self.coords1)
        x = self.nl(self.convd21(x))

        x = F.upsample_bilinear(x, scale_factor=2) # --

        x = self.coordconv(x, self.coords0)    # point
        x = self.nl(self.convd1point(x))

        x = self.ln6(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = x * self.betas[2] + x_e11 * self.alphas[2]

        x = self.coordconv(x, self.coords0)
        output = F.sigmoid(self.convd11(x))

        out, mask = output[:, :3, :, :], output[:, 3:4, :, :]

        mask = (mask + self.mlm_offset).clip(0, 1)
        # mask = dist.Bernoulli(mask).sample().to(torch.bool).expand(b, c, h, w)

        # -- The sample is the input with only the masked pixels replaced by the output pixels.
        res = out * mask + img * (1-mask)

        return res


class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, nl=torch.relu, mask_channel=False,
                 autoregressive=True, dropout=0.1, nosqrt=False):
        """

        :param emb:
        :param heads:
        :param depth:
        :param seq_length:
        :param num_tokens:
        :param nl:
        :param nosqrt: Don't use the square root in scaling the attention weights. Required for muP to work.
        :param mask_channel: Add an extra output channel (used for masking)
        """

        super().__init__()

        self.emb = emb
        self.num_tokens = num_tokens

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        self.toprobs = nn.Linear(emb, num_tokens + 1) if mask_channel else nn.Linear(emb, num_tokens)

        tblocks = []
        for _ in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=autoregressive, nl=nl,
                                 dropout=dropout, sa_kwargs={'scalefactor': 1/(emb/heads) if nosqrt else 1/math.sqrt(emb/heads) })
            )

        self.tblocks = nn.ModuleList(modules=tblocks)

    def forward(self, x, z=None):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        for i, block in enumerate(self.tblocks):
            x = block(x)

        x = self.toprobs(x)

        return x

    def mup(self, base_lr, width0, optcls=torch.optim.Adam, make_opt=True):
        """
        Implements the muP parametrization of Yang 2022. Re-inits all weights, and returns an Adam optimizer with the
        required learning rates per weight group.

        :param base_lr: Learning rate at `width = width0`. This is assumed to apply uniformly to all parameters.
        :param width0:
        :param optcls: Class for the optimizer to return (note that the current scaling applies to Adam and some variants, but not to SGD)
        :param make_opt: Create and return an muP optimizer.
        :return: A muP optimizer if requested, else nothing.
        """

        # Ratio between the current width and the width for which the base LR was tuned
        widthscale = self.emb / width0

        if make_opt:
            baseparms = []  # Parameters for which the base learning rate transfers directly
            scaleparms = [] # Parameters for which the base learning rate is scaled by 1 / fan_in

            # - Input matrices token and pos embeddings. These are not scaled.
            baseparms.extend(self.token_embedding.parameters())
            scaleparms.extend(self.pos_embedding.parameters())

        # - Trf blocks
        for block in self.tblocks:

            # layer norms. Not scaled.
            if make_opt:
                baseparms.extend(block.norm1.parameters())
                baseparms.extend(block.norm2.parameters())

            # SA weights and biases
            for lin in (block.attention.tokeys, block.attention.toqueries, block.attention.tovalues, block.attention.unifyheads):
                nn.init.normal_(lin.weight, mean=0.0, std=1/self.emb**0.5)
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, 0.0)

                # -- Note that the initialization in the paper is given as variance, where torch requires std, so we
                #    take the square root.
                # -- It's not entirely clear from the muP paper how to init the biases, but their code sets them to 0.
                #    This is also what happens in eqs 3 and 4

            if make_opt:
                scaleparms.extend(block.attention.parameters())

            # FF weights and biases
            for mod in block.ff:
                if type(mod) == nn.Linear:
                    nn.init.normal_(mod.weight, mean=0.0, std=1/self.emb**0.5)
                    if mod.bias is not None:
                        nn.init.constant_(mod.bias, val=0.0)

            if make_opt:
                scaleparms.extend(block.ff.parameters())

        # - Output head
        nn.init.normal_(self.toprobs.weight, mean=0.0, std=(5/128) ** 0.5 * 1/(self.emb * widthscale) ** 0.5 )
        # -- The multiplication by 128 is not strictly necessary for muP to work, but it stabilizes the logits for
        #    low numbers of heads
        nn.init.constant_(self.toprobs.bias, val=0.0)

        if make_opt:
            scaleparms.extend(self.toprobs.parameters())

            return optcls([
                {'params': baseparms},
                {'params': scaleparms, 'lr': base_lr / widthscale},
            ], lr=base_lr)


class ConditionalBlock(nn.Module):
    """
    Combines a self attention with a cross-attention to a conditional input
    """

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.1,
                 pos_embedding=None):
        super().__init__()

        self.sattention = SelfAttention(emb, heads=heads, mask=mask)
        self.cattention = Attention(emb, heads=heads, mask=False)

        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.norm3 = nn.LayerNorm(emb)


        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x, z):

        attended = self.sattention(x)
        x = self.norm1(attended + x)
        x = self.do(x)

        attended = self.cattention(queries=attended, keys=z, values=z)
        x = self.norm2(attended + x)
        x = self.do(x)

        fedforward = self.ff(x)
        x = self.norm3(fedforward + x)
        x = self.do(x)

        return x

class ConditionalTransformer(nn.Module):
    """
    Transformer for generating text (character by character), which is conditioned on an input sequence.
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, attention_type='default'):
        super().__init__()

        self.num_tokens = num_tokens

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        self.toprobs = nn.Linear(emb, num_tokens)

        tblocks = []
        for _ in range(depth):
            tblocks.append(
                ConditionalBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True)
            )

        self.tblocks = nn.ModuleList(modules=tblocks)

    def forward(self, x, z):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        assert x.size(0) == z.size(0), (x.size(), z.size())

        xtokens = self.token_embedding(x)
        ztokens = self.token_embedding(x)

        b, xt, e = xtokens.size()
        zt = ztokens.size(1)

        xpositions = self.pos_embedding(torch.arange(xt, device=d()))[None, :, :].expand(b, xt, e)
        x = xtokens + xpositions

        zpositions = self.pos_embedding(torch.arange(zt, device=d()))[None, :, :].expand(b, zt, e)
        z = ztokens + zpositions

        for i, block in enumerate(self.tblocks):
            x = block(x, z)

        x = self.toprobs(x)

        return x

def weights_init(model : nn.Module, init_mult_max=1.0, mask_prob_max=0.0, mup=False):
    """
    Re-initialize the weights of the given model.

    This is mostly best-effort; it will work for the models we use in this project. It checks for certain types of pytorch
    modules, and calls their reset_parameters() function.

    :param model:
    :param init_mult_max: Sets a multiplier applied to certain weights, not biases (after re-initialization). The multiplier
        is a uniform value between 0 and `init_mult_max`
    :param mask_prob_max: Sets the probability that any given element of a weight-tensor is masked out. That is, set to zero
       after re-initialization. Not applied to biases. The probability is a uniform random values between 0 and
        `mask_prob_max`.
    :return:
    """

    basewidth = 5*128

    if mup:
        model.mup(base_lr=None, width0=basewidth, make_opt=False)
        # The base_lr and width0 are only used for the optimizer

    if hasattr(model, 'alphas'):
        model.alphas = torch.bernoulli(torch.full_like(model.alphas, fill_value=0.5))
        model.betas  = torch.bernoulli(torch.full_like(model.betas, fill_value=0.5))

    # if hasattr(model, 'coords0'):
    #     mult = random.random() * 2
    #     model.coords0 *= mult
    #     model.coords1 *= mult
    #     model.coords2 *= mult

    for mod in model.modules():
        if type(mod) is nn.Linear or type(mod) is nn.Embedding or \
                type(mod) is nn.Conv2d or type(mod) is nn.ConvTranspose2d:

            init_mult = random.random() * init_mult_max
            mask_prob = random.random() * mask_prob_max

            # print(type(mod))
            # print(mod.weight.data[0])
            if not mup:
                mod.reset_parameters()

            mod.weight.data *= init_mult
            # mod.weight.data **= mask_prob

            if mask_prob > 0.0:
                mask = torch.bernoulli(torch.full_like(mod.weight.data, fill_value=mask_prob)).to(torch.bool)
                mod.weight.data[mask] = 0.0

            # print(mod.weight.data[0])
            # print()

def weights_init_plain(model : nn.Module, init_mult_max=1.0, mask_prob_max=0.0):
    """
    Re-initialize the weights of the given model.

    This is mostly best-effort; it will work for the models we use in this project. It checks for certain types of pytorch
    modules, and calls their reset_parameters() function.

    :param model:
    :param init_mult_max: Sets a multiplier applied to certain weights, not biases (after re-initialization). The multiplier
        is a uniform value between 0 and `init_mult_max`
    :param mask_prob_max: Sets the probability that any given element of a weight-tensor is masked out. That is, set to zero
       after re-initialization. Not applied to biases. The probability is a uniform random values between 0 and
        `mask_prob_max`.
    :return:
    """

    for mod in model.modules():
        if type(mod) is nn.Linear or type(mod) is nn.Embedding:

            init_mult = init_mult_max
            mask_prob = mask_prob_max

            # print(type(mod))
            # print(mod.weight.data[0])
            mod.reset_parameters()

            mod.weight.data *= init_mult

            if mask_prob > 0.0:
                mask = torch.bernoulli(torch.full_like(mod.weight.data, fill_value=mask_prob)).to(torch.bool)
                mod.weight.data[mask] = 0.0


def weights_init_minimal(model, init_mult_max):
    """
    Samples one random weight multiplier that is applied uniformly to the whole network.
    :param model:
    :param init_mult_max:
    :return:
    """

    logwm = random.random() * (math.log(init_mult_max) - 1) + 1
    wm =  math.exp(logwm)

    for mod in model.modules():
        if type(mod) is nn.Linear or type(mod) is nn.Embedding:

            mod.reset_parameters()
            mod.weight.data *= wm