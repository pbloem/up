import fire

import up
from up.util import d

import torch
from torch import nn
import torch.nn.functional as F

import tqdm
from tqdm import trange

"""
A test for a kind of hyperparametrization of the source model
"""

class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def prod(iterable):
    r = 1
    for elm in iterable:
        r *= elm
    return r

def kl(mean, logvar):
    kl = 0.5 * torch.sum(logvar.exp() - logvar + mean.pow(2) - 1)
    return kl

def slice(raw, sizes):
    """
    Slice the given vector into tensors with the shapes given by `sizes`

    :param raw: A vector. Its numel should be the same as the total number of elements
    implied by the size tuples in `sizes`.
    :param sizes: A dicts mapping names to tensor size tuples.
    :return: A dict of tensors. The keys are those in
    """
    res = {}
    fr = 0
    for name, size in sizes.items():
        tot = prod(size)
        sl = raw[fr:fr+tot]
        sl = sl.reshape(*size)
        res[name] = sl
        fr = fr+tot

    return res


def go(emb=32, bs=64, batches=500, rep=2, num_tokens=256, context=256, lr=1e-2,
       latent=256, kl_alpha=1.0, acc=3, fake_hyper=False, skip_sample=False):

    model = up.LSTMGen(emb, mask_channel=False, layers=1, num_tokens=num_tokens)

    sizes, total = {}, 0
    for name, parm in model.named_parameters():
        sizes[name] = parm.size()
        total += parm.numel()
    print('total parms in main net', total)

    # rawparms = nn.Parameter(torch.randn(size=(total,)))
    if fake_hyper: # fake hyper network that just returns the parameters
        class FH(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.randn(size=(total*2,)))

            def forward(self, x): # ignore x
                return self.p
        hyper = FH()
    else: # real hypernetwork that samples them from a genertor
        hyper = nn.Sequential(
            nn.Linear(latent, total), nn.ReLU(),
            nn.Linear(total, total), nn.ReLU(),
            nn.Linear(total, total*2)
        )

    if torch.cuda.is_available():
        hyper.cuda()
        model.cuda()

    # opt = torch.optim.Adam(lr=lr, params = model.parameters())
    opt = torch.optim.Adam(lr=lr, params=hyper.parameters())

    for i in (bar := trange(batches)):

        batch = torch.randint(high=num_tokens, size=(bs, rep), device=d())
        batch = batch.tile((1, context//rep))

        # -- Each instance is a repeating sequence of `rep` randomly chosen characters.

        input, target = batch[:, :-1], batch[:, 1:]
        # output = model(input)
        latentin = torch.randn(size=(1, latent,), device=d() )
        rawparm = hyper(latentin).squeeze(0)

        # reparamertized sample
        mean, logvar = rawparm[:total], rawparm[total:]
        if skip_sample:
            sample = mean
        else:
            eps = torch.randn((total,), device=d())
            sample = mean + (0.5 * logvar).exp() * eps

        output = torch.func.functional_call(model, slice(sample, sizes), input, strict=True)

        loss = F.cross_entropy(output.permute(0, 2, 1), target, reduction='mean')
        kl_loss = kl(mean, logvar)
        rloss = loss + kl_alpha * kl_loss

        bar.set_postfix({'l': loss.item(), 'kl' : kl_loss.item()})

        rloss.backward()

        if i % acc == 0:
            opt.step()
            opt.zero_grad()

        # print(parms.keys())
        # exit()

if __name__ == '__main__':
    fire.Fire()