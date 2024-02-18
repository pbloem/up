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
        batches_per_source=50,    # how many batches to try for each sampled source model
        wmr=(1e5, 1e7),           # range of weight multiplier (samples are taken uniformly between these two)
        logspace=False,
        emb=768,
        heads=8,
        cdepth=8,
        context=128,
        num_tokens=512,
        temperature=0.0,
        nonlinearity='relu',
        dp=False,
        type='minimal'
    ):

    # Initialize the source model
    cmp_source = up.GTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=num_tokens,
            nl=nl(nonlinearity), mask_channel=True)

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
        if type == 'minimal':
            up.weights_init_minimal(cmp_source, current_wm)
        else:
            raise

        for _ in range(batches_per_source):
            with torch.no_grad():
                # Re-initialize the parameters of source (i.e. sample a random source)
                input = torch.randint(low=0, high=num_tokens, size=(batch_size, context), device=d())

                output = cmp_source(input)

                chars, mask = output[:, :, :-1], output[:, :, -1]

                chars = sample(chars, temperature=temperature)
                mask = torch.sigmoid(mask).to(torch.bool)

                chars[mask] = input[mask]

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
    fire.Fire(go)