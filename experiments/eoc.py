import fire, random, pickle, math

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import up
from up.util import d, sample, sample_sequence, clean, ca
import json
from tqdm import trange, tqdm

from collections import Counter

def go(
        n=1000,                   # number of samples to take (points in the scatterplot
        batch_size=40,            #
        batches_per_source=1,    # how many batches to try for each sampled source model
        wmr=(1e1, 1e5),           # range of weight multiplier (samples are taken (log) uniformly between these two)
        logspace=False,
        emb=768,
        heads=8,
        cdepth=3,
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

def get_depth(width):
    """
    Compute the optimal depth for a transformer model of a given width.

    :param width:
    :return:
    """
    # Constants from the paper
    a, b = 5.039, 5.55e-2

    return int(round( (math.log(width) - a) / b ))

def rmask(tensor, prob):

    mask = torch.bernoulli(torch.full_like(tensor, fill_value=prob)).to(torch.bool)
    tensor[mask] = 0.0


def rep_sample(width=384,
                widthperhead=128,
                context = 512,
                subcontext = 64,
                nonlinearity = 'relu',
                samples = 4,
                batch_size = 4,
                temperature = 0.01,
                mult1 = 4,
                mult2 = 1,
                multb = 0,
                usemask = False,
                num_tokens=256,
                reps = 3,
                sequential = False
    ):

        if sequential:
            source = up.ConditionalTransformer(emb=width, heads=width//widthperhead, depth=get_depth(width), seq_length=context,
                        num_tokens=num_tokens)
        else:
            source = up.GTransformer(emb=width, heads=width//widthperhead, depth=get_depth(width), seq_length=context,
                        num_tokens=num_tokens, nl=nl(nonlinearity), mask_channel=True)

        if torch.cuda.is_available():
            source.cuda()

        input = torch.randint(low=0, high=num_tokens, size=(batch_size, context), device=d())

        for i in range(reps):

            if sequential:
                up.weights_init_mup_seq(source, mult1=mult1, mult2=mult2, multb=multb, mask=True)

                # -- In sequential mode, we autoregressively sample, with z as a conditional input
                #    This is very slow, but the computational patterns we expect to see are closer to those of the model
                #    we are training (which is always autoregressive).

                seed = torch.randint(low=0, high=num_tokens, size=(batch_size, 1), device=d())
                batch = sample_sequence(source, seed, max_context=subcontext, num_tokens=num_tokens, length=context,
                                        temperature=temperature,
                                        conditional=input[:, :], verbose=True)

                chars = batch[:, :-1]
            else:
                up.weights_init_mup(source, mult1=mult1, mult2=mult2, multb=multb, mask=True)

                # input = torch.full(fill_value=0, size=(batch_size, context), device=d())
                output = source(input)

                chars, mask = output[:, :, :-1], output[:, :, -1]

                chars = sample(chars, temperature=temperature)
                mask = torch.sigmoid(mask).to(torch.bool)

                if usemask:
                    chars[mask] = input[mask]

            print(i, '---')
            for seq in chars.tolist():
                print(''.join([str(s) if s < 9 else '_' for s in up.util.remap(seq, 9)][:200]))

            input = chars

def mup_sample(
        widthperhead=128,
        heads=3,
        context=512,
        nonlinearity='relu',
        samples=4,
        batch_size=4,
        temperature=1,
        mult1=1,
        mult2=2,
        multb=1,
        usemask=False,
    ):

    num_tokens = 256
    width = heads * widthperhead

    for _ in range(samples):
        # Initialize the source model
        source = up.GTransformer(emb=width, heads=heads, depth=get_depth(width), seq_length=context, num_tokens=num_tokens,
                nl=nl(nonlinearity), mask_channel=True)

        source.mup(base_lr=None, width0=None, make_opt=False)

        source.token_embedding.weight.data *= 1
        # rmask(source.token_embedding.weight.data, random.random())

        source.pos_embedding.weight.data *= 1
        # rmask(source.pos_embedding.weight.data, random.random())

        for block in source.tblocks:
            for lin in (block.attention.tokeys, block.attention.toqueries, block.attention.tovalues, block.attention.unifyheads):
                lin.weight.data *= mult1
                if lin.bias is not None:
                    lin.bias.data.normal_() * multb
                rmask(lin.weight.data, random.random())

            for mod in block.ff:
                if type(mod) == nn.Linear:
                    mod.weight.data *= mult1
                    if mod.bias is not None:
                        mod.bias.data.normal_() * multb
                    rmask(mod.weight.data, random.random())

            source.toprobs.weight.data *= mult2
            source.toprobs.bias.data.normal_() * multb


        if torch.cuda.is_available():
            source.cuda()

        input = torch.randint(low=0, high=num_tokens, size=(batch_size, context), device=d())
        # input = torch.full(fill_value=0, size=(batch_size, context), device=d())
        output = source(input)

        chars, mask = output[:, :, :-1], output[:, :, -1]

        chars = sample(chars, temperature=temperature)
        mask = torch.sigmoid(mask).to(torch.bool)

        if usemask:
            chars[mask] = input[mask]

        for seq in chars.tolist():
            print(''.join([str(s) if s < 9 else '_' for s in up.util.remap(seq, 9)][:200]))
        print('---')

def lstm_scale(lstm : nn.LSTM, weight_mult=1.0, bias_mult=1.0):

    l = lstm.num_layers

    for k in range(l):
        for wlist in lstm.all_weights:
            for w in wlist:
                w.data *= weight_mult

        for b in getattr(lstm, 'bias_ih_l'+ str(k)), getattr(lstm, 'bias_hh_l'+ str(k)):
            b.data *= bias_mult

def lstm_sample(
        emb=128,
        layers=1,
        context=512,
        samples=4,
        batch_size=4,
        temperature=1,
        mult=1,
        reps=1, # how often to feed the output back into the model
        usemask=False,
        inrep=None,
    ):

    num_tokens = 256

    class Model(nn.Module):

        def __init__(self, emb, mask_channel, layers=1):
            super().__init__()

            self.emb = emb

            self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
            self.lstm = nn.LSTM(emb, emb, num_layers=layers, batch_first=True)
            self.toprobs = nn.Linear(emb, num_tokens + 1) if mask_channel else nn.Linear(emb, num_tokens)

        def forward(self, x):
            x = self.token_embedding(x)
            x = self.lstm(x)[0]
            x = self.toprobs(x)

            return x

    for _ in range(samples):
        # Initialize the source model
        # source = up.GTransformer(emb=width, heads=heads, depth=get_depth(width), seq_length=context, num_tokens=num_tokens,
        #         nl=nl(nonlinearity), mask_channel=True)

        if inrep is None:
            chars = torch.randint(low=0, high=num_tokens, size=(batch_size, context), device=d())
        else:
            chars = torch.randint(low=0, high=num_tokens, size=(batch_size, inrep), device=d())
            nrep = int(math.ceil(context / inrep)) # how many repeats
            chars = chars.tile((1, nrep))[:, :context]

            # for seq in chars.tolist():
            #     print(''.join([str(s) if s < 9 else '_' for s in up.util.remap(seq, 9)][:200]))
            # print('---')
            # exit()

        for _ in range(reps):

            source = Model(emb, mask_channel=False, layers=layers)

            source.token_embedding.weight.data *= 1
            lstm_scale(source.lstm, mult)

            if torch.cuda.is_available():
                source.cuda()

            chars = source(chars)
            chars = sample(chars, temperature=temperature)

        for seq in chars.tolist():
            print(''.join([str(s) if s < 9 else '_' for s in up.util.remap(seq, 9)][:200]))
        print('---')

def lstm_sample_auto(
        emb=128,
        layers=(1,2),
        seedlength=32,
        context=512,
        samples=4,
        batch_size=4,
        temperature=(-1, -4),
        mult=(5, 2),
        emb_mult=1e-8,
        reps=1, # how often to feed the output back into the model
        usemask=False,
    ):

    num_tokens = 256

    for _ in range(samples):
        # Initialize the source model
        # source = up.GTransformer(emb=width, heads=heads, depth=get_depth(width), seq_length=context, num_tokens=num_tokens,
        #         nl=nl(nonlinearity), mask_channel=True)

        conds = torch.full(fill_value=0, size=(batch_size, context), device=d(), dtype=torch.long)

        for _ in range(reps):
            # the seed
            # chars = torch.randint(low=0, high=num_tokens, size=(batch_size, seedlength), device=d())
            chars = torch.randint(low=0, high=num_tokens, size=(1, seedlength), device=d()).expand(batch_size,
                                                                                                   seedlength)
            chars[1, 0] = (chars[1, 0] + 1) % num_tokens
            chars[2,16] = (chars[2, 16] + 1) % num_tokens
            chars[3,28] = (chars[3, 28] + 1) % num_tokens


            lsamp = random.randrange(*layers)
            source = up.LSTMGen(emb, mask_channel=False, layers=lsamp)

            temp_sample = 10 ** np.random.uniform(*temperature)
            mult_sample = np.random.uniform(*mult)

            print(f'mult {mult_sample:.4} \t temp {np.log10(temp_sample):.4} \t num layers {lsamp}')

            source.token_embedding.weight.data *= 1
            lstm_scale(source.lstm, mult_sample)

            if torch.cuda.is_available():
                source.cuda()

            chars = up.util.sample_sequence(model=source, seed=chars,
                                            max_context=context, num_tokens=num_tokens,
                                            length=context-chars.size(1), temperature=temp_sample, conditional=conds)

            conds = chars

        for seq in chars.tolist():
            print(''.join([str(s) if s < 9 else '_' for s in up.util.remap(seq, 9)][:200]))
        print('---')


def lstm_sample_plot(
        emb=128,
        layers=1,
        seedlength=32,
        context=512,
        samples=4,
        temperature=(-1, -4),
        mult=(0.7, 0.95),
        emb_mult=1.0,
        usemask=False,
    ):

    num_tokens = 256
    batch_size = 2

    mults = []
    hammings = []

    for _ in trange(samples):
        # Initialize the source model
        # source = up.GTransformer(emb=width, heads=heads, depth=get_depth(width), seq_length=context, num_tokens=num_tokens,
        #         nl=nl(nonlinearity), mask_channel=True)

        conds = torch.full(fill_value=0, size=(batch_size, context), device=d(), dtype=torch.long)

        # the seed
        # chars = torch.full(fill_value=0, size=(batch_size, seedlength), device=d())
        chars = torch.randint(low=0, high=num_tokens, size=(1, seedlength), device=d()).expand(batch_size, seedlength)
        chars[1,0] = (chars[1,0] + 1) % num_tokens

        source = up.LSTMGen(emb, mask_channel=False, layers=layers)
        # source.to(torch.float64)

        temp_sample = 10 ** np.random.uniform(*temperature)
        mult_sample = np.random.uniform(*mult)

        source.token_embedding.weight.data *= emb_mult
        lstm_scale(source.lstm, mult_sample)

        if torch.cuda.is_available():
            source.cuda()

        chars = up.util.sample_sequence(model=source, seed=chars,
                                        max_context=context, num_tokens=num_tokens,
                                        length=context-chars.size(1), temperature=temp_sample, conditional=conds)
        hamming = (chars[0, :] != chars[1, :]).sum().item()

        hammings.append(hamming)
        mults.append(mult_sample)

    plt.scatter(mults, hammings, alpha=0.5)
    plt.xlabel('multiplier')
    plt.ylabel('hamming distance')

    plt.savefig(f'scatter-{emb}-{emb_mult}.png')

def example(
        widthperhead=128,
        heads=12,
        context=37,
        nonlinearity='relu',
        samples=4,
        temperature=0.5,
        mult1=2,
        mult2=4,
        multb=1,
        usemask=False,
        seed=0
    ):
    """
    Used to generate the example in Figure 1.

    :param widthperhead:
    :param heads:
    :param context:
    :param nonlinearity:
    :param samples:
    :param batch_size:
    :param temperature:
    :param mult1:
    :param mult2:
    :param multb:
    :param usemask:
    :return:
    """
    torch.manual_seed(seed)

    num_tokens = 64
    width = heads * widthperhead

    # Initialize the source model
    source = up.GTransformer(emb=width, heads=heads, depth=get_depth(width), seq_length=context, num_tokens=num_tokens,
            nl=nl(nonlinearity), mask_channel=True)


    input = torch.randint(low=0, high=num_tokens, size=(1, context), device=d())
    print_batch(input, True)

    for _ in range(samples):
        up.weights_init_mup(source, mult1=mult1, mult2=mult2, multb=multb, mask=usemask)
        output = source(input)

        chars, mask = output[:, :, :-1], output[:, :, -1]

        chars = sample(chars, temperature=temperature)

        if usemask:
            mask = torch.sigmoid(mask).to(torch.bool)
            chars[mask] = input[mask]

        for seq in chars.tolist():
            # print(''.join([str(s) if s < 9 else '_' for s in up.util.remap(seq, 9)][:200]))
            print_batch(chars, True)

def test_echo(bs=4, emb=256, conn=8, num_tokens=256, context=512, temperature=1, var=0.472, reps=50, layers=1):

    model = up.ReservoirNet(emb=emb, conn=conn, num_tokens=num_tokens, init_var=var, nl=torch.tanh, layers=layers)
    lexp = model.lyapunov(gamma0=1e-12)
    print('lyapunov exponent', lexp)

    exit()
    input = torch.randint(low=0, high=num_tokens, size=(bs, context), device=d())
    for r in range(reps):
        model = up.ReservoirNet(emb=emb, conn=conn, num_tokens=num_tokens, init_var=var, nl=torch.tanh, layers=layers)
        output = model(input)

        chars = sample(output, temperature=temperature)

        if (r % 10 == 0):
            print(r, '---')
            for seq in chars.tolist():
                print(''.join([str(s) if s < 9 else '_' for s in up.util.remap(seq, 9)][:200]))
                # print_batch(chars, True)

        input = chars

def plot_lyapunov(emb=256, conn=8, num_tokens=256, max_out=8, rng=(-2,2), cache=None):

    if cache is not None:
        with open(cache, 'r') as file:
            res = json.load(file)
            xs, ys = res['xs'], res['ys']

        plt.figure()
        ax = plt.gca()

        clean(ax)
        ax.scatter(xs, ys, alpha=0.5, s=2)
        ax.set_xscale('log')

        plt.savefig('echo.png')
        plt.savefig('echo.pdf')

    else:
        xs, ys = [], []

        while True:
            var = 10.0 ** random.uniform(*rng)
            model = up.ReservoirNet(emb=emb, conn=conn, num_tokens=num_tokens, max_out=max_out, init_var=var, nl=torch.tanh)

            if torch.cuda.is_available():
                model.cuda()

            lexp = model.lyapunov(gamma0=1e-12)

            xs.append(var)
            ys.append(lexp)

            if len(xs) % 5 == 0:

                print(len(xs), ' runs completed.')

                plt.figure()
                ax = plt.gca()

                clean(ax)
                ax.scatter(xs, ys, alpha=0.5, s=2)
                ax.set_xscale('log')

                plt.savefig('echo.png')
                plt.savefig('echo.pdf')

            with open('echo.json', 'w') as file:
                json.dump(fp=file, obj={'xs':xs, 'ys':ys})


def print_batch(batch, ascii_only):

    for seq in batch:
        for c in seq:
            if ascii_only:
                print(str(chr(c+32)), end='', flush=True)
            else:
                print(cas(c), end='', flush=True)
        print()
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

def freqmodel(seq, degree):

    ctr = Counter()

    for i in range(len(seq) - degree +1):
        ctr[tuple(seq[i:i+degree])] += 1

    return ctr

def anti(vocab=[0, 1, 2, 3, 4], seed=[4, 1, 3], n=600, lam=1e-8, degree=3):
    """
    A small test of the anti-Solomonoff principle. The idea is that given a sequence model (class) C, by actively
    predicting away from it, we are sampling a sequence that is incompressible with respect to C, but that is fully
    determined from its seed given an oracle for C. This means we get a string that has, in some sense, a more interesting
    and predictable structure than C can currently understand.

    The hope is that training C on such strings, helps it to grow to greater computational complexity.

    :param vocab:
    :param seed:
    :param n:
    :param lam:
    :param degree:
    :return:
    """

    v = len(vocab)

    ctr0 = freqmodel(seed[:-1], degree - 1) if degree > 1 else None
    ctr1 = freqmodel(seed, degree)

    for _ in range(n):
        pref = tuple(seed[-degree+1:])
        if ctr0 is not None:
            probs = [ (ctr1[pref + (t,)] + lam) / (ctr0[pref] + v * lam) for t in vocab]
        else:
            probs = [(ctr1[(t,)] + lam) / (len(seed) + v * lam) for t in vocab]

        probs = np.asarray(probs)

        assert abs(sum(probs) - 1.0) < 1e-8, f'{sum(probs)} \t {probs} {ctr1}'

        # sample = np.argmin(probs)
        sample = random.choice(np.where(probs == probs.min())[0])

        seed.append(sample)

        ctr1[tuple(seed[-degree:])] += 1

        if ctr0 is not None:
            ctr0[tuple(seed[-degree:-1])] += 1

    for i, s in enumerate(seed):
        print(s, end='')
        # if i > 0 and seed[i-1] == s:
        #     print('!', end='')

    print()

def lstm_eoc(emb=256, num_tokens=256):
    """
    Try to find some kind of edge of chaos in the LSTM

    :param emb:
    :param num_tokens:
    :return:
    """

    source = up.LSTMGen(emb=emb, mask_channel=False, num_tokens=num_tokens)







if __name__ == '__main__':
    fire.Fire()