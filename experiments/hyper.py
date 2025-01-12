import fire

import up, wandb
from up.util import tic, toc, coords, d, sample, sample_sequence, gradient_norm, remap

import torch
from torch import nn
import torch.nn.functional as F

import tqdm, math
from tqdm import trange

"""
A test for a kind of hyperparametrization of the source model
"""


class LSTMGen(nn.Module):
    """
    LSTM-based generator model
    """

    def __init__(self, emb, mask_channel, layers=1, num_tokens=256, fake_hyper=False, latent=256,
                 stdmult=1e-8, meanmult=1.0, skip_sample=False, nohyper=False):
        super().__init__()

        self.emb = emb
        self.num_tokens = num_tokens
        self.latent = latent
        self.stdmult, self.meanmult = stdmult, meanmult
        self.skip_sample = skip_sample
        self.nohyper = nohyper

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.lstm = nn.LSTM(emb, emb, num_layers=layers, batch_first=True)
        self.toprobs = nn.Linear(emb, num_tokens + 1) if mask_channel else nn.Linear(emb, num_tokens)

        self.sizes, self.total = {}, 0
        for name, parm in self.lstm.named_parameters():
            self.sizes[name] = parm.size()
            self.total += parm.numel()

        print('total parms in main net', self.total)
        # for name, size in self.sizes.items():
        #     print(f'    {name} {size}')

        if fake_hyper:  # fake hypernetwork that just returns the parameters
            class FH(nn.Module):
                def __init__(self, total):
                    super().__init__()
                    self.p = nn.Parameter(torch.empty(size=(total * 2,)))
                    torch.nn.init.uniform_(self.p, -math.sqrt(1/emb), math.sqrt(1/emb))

                def forward(self, x):  # ignore x
                    return self.p

            self.hyper = FH(self.total)
        else:  # real hypernetwork that samples them from a generator
            # base parameters
            self.base = nn.Parameter(torch.empty(size=(self.total,)))
            torch.nn.init.uniform_(self.base, -math.sqrt(1 / emb), math.sqrt(1 / emb))

            # hypernet generates residual on top of the base.
            self.hyper = nn.Sequential(
                nn.Linear(latent, self.total), nn.ReLU(),
                nn.Linear(self.total, self.total*2), nn.ReLU(),
                nn.Linear(self.total*2, self.total * 2)
            )

        self.lastmean, self.lastlv = None, None

    def forward(self, x, hidden=None, return_hidden=False, z=None):
        """

        :param x:
        :param hidden:
        :param return_hidden:
        :return:
        """

        x = self.token_embedding(x)
        if self.nohyper:
            x, hidden = self.lstm(x, hidden)
            kl_loss = torch.tensor([0.0], device=d())
            div_loss = torch.tensor([0.0], device=d())
        else:
            if z is None:
                z = torch.randn(size=(1, self.latent,), device=d())

            rawparm = self.hyper(z).squeeze(0)  # hy

            # reparamertized sample
            mean, logvar = rawparm[:self.total] * self.meanmult, rawparm[self.total:] + 2 * math.log(self.stdmult)
            kl_loss = kl(mean, logvar)

            # print(mean.var().item())
            # exit()

            if self.skip_sample:
                sample = mean
            else:
                eps = torch.randn_like(mean)
                sample = mean + (0.5 * logvar).exp() * eps

            x, hidden = torch.func.functional_call(self.lstm, slice(sample + self.base, self.sizes), x, strict=True)

            if self.lastmean is None:
                div_loss = torch.tensor([0.0], device=d())
            else:
                div_loss = ((self.lastmean - mean) ** 2).sum()
                # todo: replace by proper KL div

            self.lastmean, self.lastlv = mean.detach(), logvar.detach()

        x = self.toprobs(x)

        if return_hidden:
            return x, hidden, kl_loss, div_loss
        return x, kl_loss, div_loss

    def reset_parameters(self):

        self.token_embedding.reset_parameters()
        self.lstm.reset_parameters()
        self.toprobs.reset_parameters()

    def detsig(self, n=50):
        """
        Computes the dweterminant of the covariance matrix for n samples from the hypernetwork. This is a reasonable
        indication for how spread out the samples are.

        :param n:
        :return:
        """
        t = self.total

        with torch.no_grad():
            z = torch.randn(size=(n, self.latent), device=d())
            x = self.hyper(z)

            tcov = x @ x.transpose(0, 1)
            # -- NB: The covariance matrix (X^TX) is huge, but the determinant of XX^T is the same and much smaller, so we compute that.
            # The normalization constant (1/n^dim) is too big so we omit it. We are just looking for relative changes anyway

            return torch.linalg.det(tcov)

    def sample_sequence(self, seed, max_context, num_tokens, length=600, temperature=0.5, conditional=None, verbose=False):
        """
        Sequentially samples a batch of sequences from the model, token by token. Since the model is an RNN, we can do
        this in linear time by remembering the hidden state.

        :param model:
        :param seed: The sequence to start with.
        :param length: The total number of characters to sample.
        :param temperature: The sampling temperature.
        :param verbose: If true, the sampled sequence is also printed as it is sampled.

        :return: The sampled sequence, including the seed.
        """

        (b, seedlen) = seed.size()

        sequence = seed.detach().clone()

        # sequence = sequence[None, :].expand(batch_size, b)
        rng = trange if verbose else range

        # run the seed through the model to get the first hidden state
        # -- the last character of the seed is handled in the loop
        _, hidden = self(sequence[:, :-1], conditional[:, :sequence.size(1)-1], return_hidden=True, hidden=None)

        for i in rng(length):

            # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
            input = sequence[:, -max_context:]
            # input = sequence

            b, l = input.size()

            # Run the current input through the model

            to = seedlen + i
            output, hidden = self(input[:, -1:], conditional[:, to-1:to], return_hidden=True, hidden=hidden)

            # Sample the next token from the probabilities at the last position of the output.
            cs = sample(output[:, -1, :], temperature)
            assert cs.size() == (b,)

            sequence = torch.cat([sequence, cs[:, None]], dim=-1) # Append the sampled token to the sequence

        return sequence


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


def go(emb=32, bs=64, batches=500, rep=2, num_tokens=256, context=256, lr=3e-4,
       latent=256, kl_alpha=1.0, b_alpha=1.0, acc=3, fake_hyper=False, skip_sample=False, stdmult=1e-8, nohyper=False, meanmult=1.0,
       warmup=5_000, cooldown=-1, gc=1.0, name='hyper-test', project='hyper-test', debug=False, div_alpha=1.0):

    wd = wandb.init(
        name=name,
        project=project,
        config=locals(),
        mode= 'disabled' if debug else 'online'
    )

    model = LSTMGen(emb, mask_channel=False, layers=1, num_tokens=num_tokens, fake_hyper=fake_hyper, latent=latent,
                       skip_sample=skip_sample, meanmult=meanmult, stdmult=stdmult, nohyper=nohyper)

    if torch.cuda.is_available():
        model.cuda()

    # opt = torch.optim.Adam(lr=lr, params = model.parameters())
    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    if warmup > 0:
        # warmup = warmup / accumulate
        # sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / model_batch_size), 1.0))
        for g in opt.param_groups:
            g['max_lr'] = g['lr']
            g['lr_delta'] = g['lr'] / warmup

            g['lr'] = 0.0

    cooldown_rate = 0.5 ** (1/cooldown)
    # -- The cooldown rate is given in the number of instances to halve the learning rate over. This is the resulting
    #    multiplier per instance.
    last_cooldown = warmup

    instances_seen = 0
    for i in (bar := trange(batches)):

        batch = torch.randint(high=num_tokens, size=(bs, rep), device=d())
        batch = batch.tile((1, context//rep))

        # -- Each instance is a repeating sequence of `rep` randomly chosen characters.

        input, target = batch[:, :-1], batch[:, 1:]
        output, kl_loss, div_loss = model(input)

        loss = F.cross_entropy(output.permute(0, 2, 1), target, reduction='mean')
        bloss = 0.0 if fake_hyper else model.base.norm(p=2) # pull the base params to the origin

        rloss = loss + kl_alpha * kl_loss + b_alpha * bloss - div_alpha * div_loss

        bar.set_postfix({'l': loss.item(), 'kl' : kl_loss.item()})

        rloss.backward()

        if i % acc == 0:
            gn = gradient_norm(model)
            if gc > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gc)

            opt.step()
            opt.zero_grad()

            ds = model.detsig()

            ### Admin
            wandb.log({
                'loss': loss.item(),
                'kl': kl_loss.item(),
                'gradient norm': gn,
                'lr': opt.param_groups[0]['lr'],
                'b reg': bloss,
                'div': div_loss,
                'detsig': ds,
            }, step=instances_seen, commit=True)

        instances_seen += bs

        if warmup > 0 and instances_seen <= warmup:
            for g in opt.param_groups:
                if g['lr'] < g['max_lr']:
                    g['lr'] += g['lr_delta'] * batch.size(0)

        if cooldown > 0 and instances_seen > warmup:
            for g in opt.param_groups:
                g['lr'] *= cooldown_rate ** batch.size(0)

        # print(parms.keys())
        # exit()

if __name__ == '__main__':
    fire.Fire()