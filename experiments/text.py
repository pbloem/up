import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sklearn
from sklearn.decomposition import PCA

from former import TransformerBlock, SelfAttention, Attention

import up
from up.util import Reshape, kl_loss, vae_sample, gradient_norm, coords, markov
from up.data import load_data, cas, to_str, to_bytes
from up import VAE, AE, ConditionalBlock, ConditionalTransformer, GTransformer

from former.util import d, here, tic, toc, sample_batch, enwik8_string, enwik8_bytes, estimate_compression
import former

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from argparse import ArgumentParser
import wandb

import random, sys, math, gzip, time, os

from tqdm import tqdm, trange

from collections import Counter

import fire
import copy

# The model predict a single byte. We usually interpret this as ASCII, skipping the non-printable characters
NUM_TOKENS = 256
# Used for converting between nats and bits
LOG2E = math.log2(math.e)

def sample_sequence(model, seed, max_context, num_tokens, length=600, temperature=0.5, conditional=None):
    """
    Sequentially samples a batch of sequences from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.

    :return: The sampled sequence, including the seed.
    """

    (b, l) = seed.size()

    sequence = seed.detach().clone()

    # sequence = sequence[None, :].expand(batch_size, b)

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[:, -max_context:]

        b, l = input.size()

        # Run the current input through the model
        output = model(input) if conditional is None else model(input, conditional)
        # output = F.log_softmax(output, dim=-1)

        assert output.size() == (b, l, num_tokens)

        # Sample the next token from the probabilitys at the last position of the output.
        cs = sample(output[:, -1, :], temperature)

        assert cs.size() == (b,)

        # if verbose:
        #     print(str(chr(max(32, c))), end='', flush=True)

        # print(sequence.size(), cs.size())
        # exit()

        sequence = torch.cat([sequence, cs[:, None]], dim=-1) # Append the sampled token to the sequence

    return sequence

def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax(dim=-1)

    p = F.softmax(lnprobs / temperature, dim=-1)
    cd = dist.Categorical(p)

    return cd.sample()

def weights_init(model : nn.Module, init_mult_max=1.0, mask_prob_max=0.0):

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

def num_tokens(ascii_only):
    return (128-32) if ascii_only else 256

def go(emb=768, heads=8, cdepth=3, mdepth=6, context=128, temperature=0.5, sample_batch_size=100, model_batch_size=200,
         buffer_size=2000,
         reset_prob=0.01, num_batches=100_000, lr=3e-4, tags=[],
         debug=False, warmup=100_000, eval_every=5_000, print_every=500, gc=1.0,
         sequential=False, eval_samples=10_000, steps_per_sample=1, mlm_prob=0.15, ascii_only=False,
         init_mult=1.0
       ):
    """
    Generates a dataset by sampling sequences autoregressively from a given model.

    Separates model and source. Samples over all depths of universal model.

    """

    wd = wandb.init(
        project='prior',
        tags=tags,
        config=locals(),
        mode= 'disabled' if debug else 'online'
    )

    datasets = {
        'dyck' : torch.tensor(load_data('dyck'), dtype=torch.long),
        'ndfa' : torch.tensor(load_data('ndfa'), dtype=torch.long),
        'toy'  : torch.tensor(load_data('toy'),  dtype=torch.long)
    }

    if not ascii_only:
        datasets['wp'] = torch.tensor(load_data('wp'),   dtype=torch.long)
        # -- the wp benchmark requires a full character set


    # Computation source
    cmp_source = \
        ConditionalTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=num_tokens(ascii_only)) \
        if sequential else \
        GTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=num_tokens(ascii_only))

    # Target for training
    model  = GTransformer(emb=emb, heads=heads, depth=mdepth, seq_length=context, num_tokens=num_tokens(ascii_only))

    if torch.cuda.is_available():
        cmp_source.cuda()
        model.cuda()

    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    if warmup > 0:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / model_batch_size), 1.0))

    buffer = torch.randint(low=0, high=num_tokens(ascii_only), size=(buffer_size, context), device=d())

    for i in (bar := trange(num_batches)):

        opt.zero_grad()

        tic()
        with torch.no_grad():
            # Re-initialize the parameters of source (i.e. sample a random source)
            weights_init(cmp_source, init_mult=init_mult)

            # slice a random selection of rows from the buffer (without replacement)
            iz = random.sample(range(buffer_size), sample_batch_size)
            z = buffer[iz, :]

            # replace some random rows with uniform random characters
            rows = torch.bernoulli(torch.full(size=(sample_batch_size, 1), fill_value=reset_prob))
            mask = rows.expand(sample_batch_size, context).to(torch.bool)

            uniform = torch.randint(low=0, high=num_tokens(ascii_only), size=(sample_batch_size, context), device=d())
            z[mask] = uniform[mask]

            # pass it through a randomly chosen model
            if sequential:
                # -- In sequential mode we autoregressively sample, with z as a conditional input
                #    This is very slow, but the computational patterns we expect to see are closer to those of the model
                #    we are training (which is always autoregressive).

                seed = torch.randint(low=0, high=num_tokens(ascii_only), size=(sample_batch_size, 1), device=d())
                batch = sample_sequence(cmp_source, seed, context, num_tokens=num_tokens(ascii_only), length=context,
                                     temperature=temperature,
                                     conditional=z)

                buffer[iz, :] = batch[:, :-1]

            else:
                # -- In non-sequential mode, we follow the MLM strategy. We sample output positions with probability
                #    `mlm_prob` and replace these positions in the batch by the ouput of cmp(batch). The remainder is
                #    kept the same as the input and the batch is place back into the buffer.
                #
                #    At mlm_prob=1.0, this results in a fully new random sequence sampled. The idea of lower values is
                #    that this results in more internal correlation in the samples. That is, the value of one token can
                #    be inferred from other parts of the sequence more easily.

                output = cmp_source(z)
                output = sample(output, temperature=temperature)

                # mask out random columns for the model to replace
                rows = torch.bernoulli(torch.full(size=(1, context), fill_value=mlm_prob))
                mask = rows.expand(sample_batch_size, context).to(torch.bool)

                z[mask] = output[mask]

                buffer[iz, :] = z

            # -- The output of sample_sequence is context + 1 because of the seed, so we slice off the last character. The
            #    seed is likely more important in the long run

        sampletime = toc()

        tic()
        # Perform n training steps on batches samples from the buffer
        for _ in range(steps_per_sample):
            iz = random.sample(range(buffer_size), model_batch_size)
            batch = buffer[iz, :]

            input  = batch[:, :-1]
            target = batch[:, 1:]

            output = model(input)
            loss = F.cross_entropy(output.transpose(2, 1), target)

            loss.backward()
            opt.step()
        traintime = toc()

        wandb.log({
            'loss': loss,
            'learning_rate': sch.get_last_lr()[0],
            'gradient_norm': gradient_norm(model),
            'sample_time': sampletime,
            'train_time': traintime
        })
        bar.set_postfix({'loss': f'{loss:.02}'})

        if warmup > 0:
              sch.step()


        if i % print_every == 0:

            print('target')
            print_batch(batch[:4, :], ascii_only)

            print('model output')

            seed = torch.randint(low=0, high=num_tokens(ascii_only), size=(4, 1), device=d())
            output = sample_sequence(model, seed, context, num_tokens = num_tokens(ascii_only), length=context,
                                            temperature = temperature)
            print_batch(output, ascii_only)

        if eval_every > 0 and i % eval_every == 0:

            for name, data in datasets.items():
                print(f'evaluating {name}')

                with torch.no_grad():
                    est = estimate_compression(
                        model=model,
                        data=data,
                        nsamples=eval_samples,
                        context=context,
                        batch_size=model_batch_size * 2,
                        model_produces_logits=True
                    )

                wandb.log({f'val-{name}': est})

def print_batch(batch, ascii_only):

    for seq in batch:
        for c in seq:
            if ascii_only:
                print(str(chr(c+32)), end='', flush=True)
            else:
                print(cas(c), end='', flush=True)
        print()
    print()


def run_every(emb=768, heads=8, cdepth=3, context=128, temperature=0.5,
         buffer_size=20,
         reset_prob=0.01, max_depth=30, tags=[],
         debug=False, warmup=100_000, eval_every=5_000, print_every=500, gc=1.0,
         sequential=False, eval_samples=10_000, steps_per_sample=1, mlm_prob=0.15, ascii_only=False,
         init_mult_max=1.0, mask_prob_max=1.0
       ):
    """
    Samples from the "universal" distribution at every n.
    """

    wd = wandb.init(
        project='prior',
        tags=tags,
        config=locals(),
        mode= 'disabled' if debug else 'online'
    )

    # Computation source
    cmp_source = \
        ConditionalTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=num_tokens(ascii_only)) \
        if sequential else \
        GTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=num_tokens(ascii_only), mask_channel=True)

    if torch.cuda.is_available():
        cmp_source.cuda()

    buffer = torch.randint(low=0, high=num_tokens(ascii_only), size=(buffer_size, context), device=d())

    for n in range(max_depth):

        print(n)
        print_batch(buffer, ascii_only)
        print()

        for i in trange(buffer_size):

            with torch.no_grad():

                # Re-initialize the parameters of source (i.e. sample a random source)
                weights_init(cmp_source, init_mult_max=init_mult_max, mask_prob_max=mask_prob_max)

                z = buffer[i:i+1]

                # pass it through a randomly chosen model
                if sequential:
                    # -- In sequential mode we autoregressively sample, with z as a conditional input
                    #    This is very slow, but the computational patterns we expect to see are closer to those of the model
                    #    we are training (which is always autoregressive).

                    seed = torch.randint(low=0, high=num_tokens(ascii_only), size=(sample_batch_size, 1), device=d())
                    batch = sample_sequence(cmp_source, seed, context, num_tokens=num_tokens(ascii_only), length=context,
                                         temperature=temperature,
                                         conditional=z)

                    buffer[i, :] = batch[:, :-1]

                else:
                    # -- In non-sequential mode, we follow the MLM strategy. We sample output positions with probability
                    #    `mlm_prob` and replace these positions in the batch by the ouput of cmp(batch). The remainder is
                    #    kept the same as the input and the batch is place back into the buffer.
                    #
                    #    At mlm_prob=1.0, this results in a fully new random sequence sampled. The idea of lower values is
                    #    that this results in more internal correlation in the samples. That is, the value of one token can
                    #    be inferred from other parts of the sequence more easily.

                    output = cmp_source(z)
                    chars, mask = output[:, :, :-1], output[:, :, -1]

                    chars = sample(chars, temperature=temperature)
                    # mask out random columns for the model to replace

                    mask = torch.sigmoid(mask).round()
                    mask = mask.to(torch.bool)

                    z[mask] = chars[mask]

                    buffer[i, :] = z

                # -- The output of sample_sequence is context + 1 because of the seed, so we slice off the last character. The
                #    seed is likely more important in the long run



def run_sample(emb=768, heads=8, cdepth=3, context=128, temperature=0.5, sample_batch_size=100,
         buffer_size=2000,
         reset_prob=0.01, num_batches=100_000, tags=[],
         debug=False, warmup=100_000, eval_every=5_000, print_every=500, gc=1.0,
         sequential=False, eval_samples=10_000, steps_per_sample=1, mlm_prob=0.15, ascii_only=False,
         init_mult_max=1.0
       ):
    """
    Generates a dataset by sampling sequences autoregressively from a given model.

    Separates model and source. Samples over all depths of universal model.
    """

    wd = wandb.init(
        project='prior',
        tags=tags,
        config=locals(),
        mode= 'disabled' if debug else 'online'
    )

    # Computation source
    cmp_source = \
        ConditionalTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=num_tokens(ascii_only)) \
        if sequential else \
        GTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=num_tokens(ascii_only))

    if torch.cuda.is_available():
        cmp_source.cuda()

    buffer = torch.randint(low=0, high=num_tokens(ascii_only), size=(buffer_size, context), device=d())

    for i in (bar := trange(num_batches)):

        with torch.no_grad():
            # Re-initialize the parameters of source (i.e. sample a random source)
            weights_init(cmp_source, init_mult_max=init_mult_max)

            # slice a random selection of rows from the buffer (without replacement)
            iz = random.sample(range(buffer_size), sample_batch_size)
            z = buffer[iz, :]

            # replace some random rows with uniform random characters
            rows = torch.bernoulli(torch.full(size=(sample_batch_size, 1), fill_value=reset_prob))
            mask = rows.expand(sample_batch_size, context).to(torch.bool)

            uniform = torch.randint(low=0, high=num_tokens(ascii_only), size=(sample_batch_size, context), device=d())
            z[mask] = uniform[mask]

            # pass it through a randomly chosen model
            if sequential:
                # -- In sequential mode we autoregressively sample, with z as a conditional input
                #    This is very slow, but the computational patterns we expect to see are closer to those of the model
                #    we are training (which is always autoregressive).

                seed = torch.randint(low=0, high=num_tokens(ascii_only), size=(sample_batch_size, 1), device=d())
                batch = sample_sequence(cmp_source, seed, context, num_tokens=num_tokens(ascii_only), length=context,
                                     temperature=temperature,
                                     conditional=z)

                buffer[iz, :] = batch[:, :-1]

            else:
                # -- In non-sequential mode, we follow the MLM strategy. We sample output positions with probability
                #    `mlm_prob` and replace these positions in the batch by the ouput of cmp(batch). The remainder is
                #    kept the same as the input and the batch is place back into the buffer.
                #
                #    At mlm_prob=1.0, this results in a fully new random sequence sampled. The idea of lower values is
                #    that this results in more internal correlation in the samples. That is, the value of one token can
                #    be inferred from other parts of the sequence more easily.

                output = cmp_source(z)
                output = sample(output, temperature=temperature)

                # mask out random columns for the model to replace
                rows = torch.bernoulli(torch.full(size=(1, context), fill_value=mlm_prob))
                mask = rows.expand(sample_batch_size, context).to(torch.bool)

                z[mask] = output[mask]

                buffer[iz, :] = z

            # -- The output of sample_sequence is context + 1 because of the seed, so we slice off the last character. The
            #    seed is likely more important in the long run

        if i % print_every == 0:

            # iz = random.sample(range(buffer_size), 4)
            # z = buffer[iz, :]

            print_batch(buffer[:4, :], ascii_only)
            print()

def nonseq(emb=768, heads=8, depth=12, context=128, temperature=0.5, batch_size=256, num_batches=100_000, lr=3e-4, tags=[],
           debug=False, warmup=100_000, pretrain_its=1, eval_every=5_000, gc=1.0):
    """
    Generates a dataset by sampling sequences autoregressively from a given model.

    :param emb:
    :param heads:
    :param depth:
    :param context:
    :param temperature:
    :param batch_size:
    :param num_instances:
    :param model_file: If none, the model is randomly initialized, and re-initialized for every batch. Otherwise, the
    weights are kept fixed, but dropout is left on to ensure variety. TODO: add some Gaussian noise with reparametrization.
    :return:
    """

    wd = wandb.init(
        project='prior',
        tags=tags,
        config=locals(),
        mode= 'disabled' if debug else 'online'
    )

    # Randomness source (uniform if None)
    rnd_source = None
    # Computation source
    cmp_source = GTransformer(emb=emb, heads=heads, depth=depth, seq_length=context, num_tokens=NUM_TOKENS)
    # Target for training
    model  = GTransformer(emb=emb, heads=heads, depth=depth, seq_length=context, num_tokens=NUM_TOKENS)

    if torch.cuda.is_available():
        cmp_source.cuda()
        model.cuda()

    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    if warmup > 0:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / batch_size), 1.0))

    for p in range(pretrain_its):
        for i in (bar := trange(num_batches)):

            opt.zero_grad()

            # Re-initialize the parameters (i.e. sample a random source)
            weights_init(cmp_source)

            with torch.no_grad():
                # Sample a uniform random sequence of tokens.
                uniform = torch.randint(low=0, high=NUM_TOKENS, size=(batch_size, context), device=d())

                # In the first iteration, we feed the source uniform random tokens. In the higher generations, we sample
                # inputs from the previous random model.
                if rnd_source is None:
                    source_in = uniform
                else:
                    source_in = rnd_source(uniform)
                    source_in = source_in.argmax(dim=-1)

                source_out = cmp_source(source_in)

            output = model(source_in)
            loss = F.cross_entropy(output.transpose(2, 1), F.softmax(source_out, dim=-1).transpose(2, 1))

            loss.backward()
            opt.step()
            training_time = toc()

            wandb.log({'loss': loss, 'learning_rate': sch.get_last_lr()[0], 'gradient_norm': gradient_norm(model)})
            bar.set_postfix({'loss': f'{loss:.02}'})

            if warmup > 0:
                  sch.step()

            if i % eval_every == 0:

                for _ in range(4):
                    # Print some samples (just argmax for now)
                    with torch.no_grad():
                        # Sample a uniform random sequence of tokens.
                        uniform = torch.randint(low=0, high=NUM_TOKENS, size=(3, context), device=d())

                        # In the first iteration, we feed the source uniform random tokens. In the higher generations, we sample
                        # inputs from the previous random model.
                        if rnd_source is None:
                            source_in = uniform
                        else:
                            source_in = rnd_source(uniform)
                            source_in = source_in.argmax(dim=-1)

                        source_out = cmp_source(source_in)

                    print('target')

                    for seq in sample(source_out, temperature=temperature):
                        for c in seq:
                            print(str(chr(c + 32)), end='', flush=True)
                        print()
                    print()

                    print('model output')

                    output = model(source_in)

                    for seq in sample(output, temperature=temperature):
                        for c in seq:
                            print(str(chr(c + 32)), end='', flush=True)
                        print()
                    print()

        # Take a snapshot of the current model
        rnd_source = copy.deepcopy(model)

        # -- We just keep on training `model`. Think of the previous universal model as a kind of "initialization" for the
        #    next one

def toy(name='dyck', emb=768, heads=8, depth=12, context=128, temperature=0.5, batch_size=128, num_batches=10_000, lr=3e-4,
           debug=False, warmup=100_000, eval_every=5_000, tags=[], gc=1.0):

    wd = wandb.init(
        project='prior',
        tags=tags,
        config=locals(),
        mode= 'disabled' if debug else 'online'
    )

    sequences, (i2t, t2i) = up.data.load_toy(name='ndfa')
    random.shuffle(sequences)

    # Add start/end
    sequences = [ [t2i['.start']] + s + [t2i['.end']] for s in sequences]

    # Flatten
    sequences = [t for s in sequences for t in s]

    sequences = torch.tensor(sequences)

    model = GTransformer(emb=emb, heads=heads, depth=depth, seq_length=context, num_tokens=len(i2t))
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr = lr, params=model.parameters())
    if warmup > 0:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / batch_size), 1.0))

    for _ in (bar := trange(num_batches)):

        opt.zero_grad()

        source, target = sample_batch(sequences, context, batch_size)

        # for seq in source:
        #     print(''.join(i2t[i] for i in seq))
        # exit()

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        output = model(source)
        loss = F.cross_entropy(output.transpose(2, 1), target)

        loss.backward()
        if gc > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gc)

        opt.step()
        if warmup > 0:
            sch.step()

        wandb.log({
            'loss': loss,
            'learning_rate': sch.get_last_lr()[0],
            'gradient_norm': gradient_norm(model)
        })
        bar.set_postfix({'loss': f'{loss:.02}'})

def test():
    pass
    # data = enwik8_bytes()
    # data = data[0] + data[1] + data[2]
    #
    # print(len(data))
    # print(data[11000:11100])
    # print(data[12000:12100])
    # print(data[13000:13100])
    #
    # ctr = Counter()
    #
    # for ch in data:
    #     ctr[ch] += 1
    #
    # print(len(ctr))
    # print(list(str(chr(ch)) for ch, _ in ctr.most_common()))

    # for i in range(NUM_TOKENS):
    #     print(i, cas(i) )

    # tic()
    # train, test = to_str(load_data(name='wp-train')), to_str(load_data(name='wp-test'))
    # # train, test = '11001100100100101000111011010010', '00'
    # print(f'loaded ({toc():.4}s).')
    #
    # m = markov(train, test, max_order=5, verbose=True)
    #
    # print(m)


if __name__ == "__main__":

    fire.Fire()
