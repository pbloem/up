import fire, random, pickle, math

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import up
from up.util import d, sample

from tqdm import trange

def go(
        n=1000,                   # number of samples to take (points in the scatterplot
        batch_size=40,            #
        batches_per_source=1,    # how many batches to try for each sampled source model
        wmr=(1e5, 1e7),           # range of weight multiplier (samples are taken uniformly between these two)
        logspace=False,
        emb=768,
        heads=8,
        cdepth=8,
        its=1,                    # How often to feed the model's output back to itself.
        context=128,
        num_tokens=512,
        temperature=0.0,
        nonlinearity='relu',
        dp=False,
        type='minimal',
        skip_mask=False,
        bidirectional=False
    ):

    # Initialize the source model
    cmp_source = up.GTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=num_tokens,
            nl=nl(nonlinearity), mask_channel=True, autoregressive=not bidirectional)

    if torch.cuda.is_available():
        cmp_source.cuda()

    if dp:
        cmp_source = torch.nn.DataParallel(cmp_source)

    results = []
    for _ in trange(n):

        if logspace:
            lower, upper = math.log(wmr[0]), math.log(wmr[1])
            current_wm = random.random() * (upper-lower) + lower
            current_wm = math.exp(current_wm)
        else:
            current_wm = random.random() * (wmr[1] - wmr[0]) + wmr[0]

        seq = []
        if type == 'default':
            up.weights_init(cmp_source, current_wm)
        elif type == 'plain':
            up.weights_init_plain(cmp_source, current_wm)
        elif type == 'minimal':
            up.weights_init_minimal(cmp_source, current_wm)
        elif type == 'input':
            cmp_source.token_embedding.weight.data *= current_wm
            cmp_source.pos_embedding.weight.data *= current_wm
        else:
            raise

        for _ in range(batches_per_source):
            with torch.no_grad():
                # Re-initialize the parameters of source (i.e. sample a random source)
                input = torch.randint(low=0, high=num_tokens, size=(batch_size, context), device=d())

                for _ in range(its):
                    output = cmp_source(input)

                    chars, mask = output[:, :, :-1], output[:, :, -1]

                    chars = sample(chars, temperature=temperature)
                    mask = torch.sigmoid(mask).to(torch.bool)

                    if not skip_mask:
                        chars[mask] = input[mask]

                    input = chars

                for instance in chars.tolist():
                    seq.extend(instance)

        # print(current_wm, up.util.remap(seq)[:32])
        # print()

        results.append( (current_wm, up.util.measure_gzip(seq) / (batches_per_source * batch_size * context)) )

    results.sort()
    for w, b in results:
        print(w,'\t', b)

    with open('result.pkl', 'wb') as file:
        pickle.dump(results, file)

    results = np.asarray(results)
    plt.figure(figsize=(16,9))
    plt.scatter(results[:, 0], results[:, 1], s=8, alpha=0.5)
    plt.gca().set_xscale('log')

    plt.savefig('results.png')


def printseqs(
        n=10,                     # number of samples to take (points in the scatterplot
        batch_size=40,            #
        wmr=(1e5, 1e7),           # range of weight multiplier (samples are taken uniformly between these two)
        logspace=False,
        emb=768,
        heads=8,
        cdepth=8,
        its=1,                    # How often to feed the model's output back to itself.
        context=128,
        num_tokens=512,
        temperature=0.0,
        nonlinearity='relu',
        dp=False,
        type='minimal',
        skip_mask=False
    ):

    # Initialize the source model
    cmp_source = up.GTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=num_tokens,
            nl=nl(nonlinearity), mask_channel=True)

    if torch.cuda.is_available():
        cmp_source.cuda()

    if dp:
        cmp_source = torch.nn.DataParallel(cmp_source)

    if not logspace:
        wms = np.linspace(wmr[0], wmr[1], num=n)
    else:
        wms = np.linspace(np.log(wmr[0]), np.log(wmr[1]), num=n)
        wms = np.exp(wms)

    for current_wm in wms:

        if type == 'default':
            up.weights_init(cmp_source, current_wm)
        elif type == 'plain':
            up.weights_init_plain(cmp_source, current_wm)
        elif type == 'minimal':
            up.weights_init_minimal(cmp_source, current_wm)
        elif type == 'input':
            cmp_source.token_embedding.weight.data *= current_wm
            cmp_source.pos_embedding.weight.data *= current_wm
        else:
            raise

        print(current_wm)
        for _ in range(3):
            seq = []
            with torch.no_grad():
                # Re-initialize the parameters of source (i.e. sample a random source)
                input = torch.randint(low=0, high=num_tokens, size=(batch_size, context), device=d())

                for _ in range(its):
                    output = cmp_source(input)

                    chars, mask = output[:, :, :-1], output[:, :, -1]

                    chars = sample(chars, temperature=temperature)
                    mask = torch.sigmoid(mask).to(torch.bool)

                    if not skip_mask:
                        chars[mask] = input[mask]

                    input = chars

                for instance in chars.tolist():
                    seq.extend(instance)

                print(''.join(str(s) for s in  up.util.remap(seq, 9)))
        print()

def nl(name : str):
    if name == 'relu':
        return torch.relu

    if name == 'sign':
        return torch.sign

    if name == 'sigmoid':
        return torch.sigmoid

    if name.startswith('sigmoid'):
        temp = int(name[7])
        return lambda x : torch.sigmoid(x * 10**-temp)

    raise Exception(f'Nonlinearity {name} not recognized.')

if __name__ == '__main__':
    fire.Fire()