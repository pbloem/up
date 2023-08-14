
import up
from up.util import tic, toc, coords, d, sample, sample_sequence, gradient_norm
from up.data import load_data, cas

from former.util import d, here, tic, toc, sample_batch, enwik8_string, enwik8_bytes, estimate_compression
import former

import wandb, random, fire

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import trange

"""
Experiment 1: We train a GPT-style transformer on the Hutter prize data (100MB of wikpidia text), and test the influence
 of universal pretraining.

"""

NUM_TOKENS = 256

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
        temp = int(name[6])
        return lambda x : torch.sigmoid(x * 10**-temp)

    raise Exception(f'Nonlinearity {name} not recognized.')

def go(emb=768, heads=8, cdepth=3, mdepth=6, context=128, temperature=0.5, sample_batch_size=100,
         buffer_size=2000, pre_batches=0, model_batch_size=None,
         reset_prob=0.01, num_batches=10_000_000, lr=3e-4, tags=[],
         debug=False, warmup=100_000, eval_every=5_000, print_every=500, gc=1.0,
         sequential=False, eval_samples=10_000, steps_per_sample=1, mlm_prob=0.15, ascii_only=False,
         init_mult_max=5.0, mask_prob_max=0.7, nonlinearity='relu', skip_eval=False, eval_ood=False):

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

    if eval_ood:
        datasets = {
            'dyck' : torch.tensor(load_data('dyck'), dtype=torch.long),
            'ndfa' : torch.tensor(load_data('ndfa'), dtype=torch.long),
            'toy'  : torch.tensor(load_data('toy'),  dtype=torch.long),
            'wp'   : torch.tensor(load_data('wp-val'), dtype=torch.long)
        }
    else:
        datasets = {
            'wp'   : torch.tensor(load_data('wp-val'), dtype=torch.long)
        }

    # Computation source
    cmp_source = \
        up.ConditionalTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=NUM_TOKENS) \
        if sequential else \
        up.GTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=NUM_TOKENS, nl=nl(nonlinearity))

    # Target for training
    model  = up.GTransformer(emb=emb, heads=heads, depth=mdepth, seq_length=context, num_tokens=NUM_TOKENS)

    if torch.cuda.is_available():
        cmp_source.cuda()
        model.cuda()

    # Throughput test to find batch size
    dummy_input  = torch.randint(low=0, high=NUM_TOKENS, size=(1, context), dtype=torch.long, device=d())
    def dummy_loss(output):
        b, c, e = output.size()
        dummy_target = torch.randint(low=0, high=NUM_TOKENS, size=(b, c), dtype=torch.long, device=d())
        return F.cross_entropy(output.transpose(1,2), dummy_target)

    if model_batch_size is None:
        print('Starting throughput test.'); tic()
        model_batch_size, batch_sizes, throughputs = up.util.find_batch_size(model=model, loss=dummy_loss, input=dummy_input, burn_in=10, samples=20, wandb=wandb)

        print(f'Finished ({toc():.4}s). Optimal batch size: {model_batch_size}. Batch sizes tested {batch_sizes}, with throughput {throughputs}.')

    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    if warmup > 0:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / model_batch_size), 1.0))

    buffer = torch.randint(low=0, high=NUM_TOKENS, size=(buffer_size, context), device=d())

    # Pretraining batches
    if pre_batches > 0:

        print('Start pre-training')

        for i in (bar := trange(pre_batches)):

            if eval_every > 0 and i % eval_every == 0 and not skip_eval:

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

            opt.zero_grad()

            tic()
            with torch.no_grad():
                # Re-initialize the parameters of source (i.e. sample a random source)
                up.weights_init(cmp_source, init_mult_max=init_mult_max, mask_prob_max=mask_prob_max)

                # slice a random selection of rows from the buffer (without replacement)
                iz = random.sample(range(buffer_size), sample_batch_size)
                z = buffer[iz, :]

                # replace some random rows with uniform random characters
                rows = torch.bernoulli(torch.full(size=(sample_batch_size, 1), fill_value=reset_prob))
                mask = rows.expand(sample_batch_size, context).to(torch.bool)

                uniform = torch.randint(low=0, high=NUM_TOKENS, size=(sample_batch_size, context), device=d())
                z[mask] = uniform[mask]

                # pass it through a randomly chosen model
                if sequential:
                    # -- In sequential mode we autoregressively sample, with z as a conditional input
                    #    This is very slow, but the computational patterns we expect to see are closer to those of the model
                    #    we are training (which is always autoregressive).

                    seed = torch.randint(low=0, high=NUM_TOKENS, size=(sample_batch_size, 1), device=d())
                    batch = sample_sequence(cmp_source, seed, context, num_tokens=NUM_TOKENS, length=context,
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
                'train_time': traintime,
                'pre-training': 1.0

            })
            bar.set_postfix({'loss': f'{loss:.02}'})

            if warmup > 0:
                  sch.step()

            if i % print_every == 0:

                print('target')
                print_batch(batch[:4, :], ascii_only)

                print('model output')

                seed = torch.randint(low=0, high=NUM_TOKENS, size=(4, 1), device=d())
                output = sample_sequence(model, seed, context, num_tokens = NUM_TOKENS, length=context,
                                                temperature = temperature)
                print_batch(output, ascii_only)

    # Fine-tuning
    print('Start finetuning')

    if warmup > 0:
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / model_batch_size), 1.0))
    # -- We keep the same optimizer (the statistics may well carry over from pre-training to finetuning)
    #    but we reset the learning rate warmup.

    data = torch.tensor(up.data.load_data('wp-train'), device='cpu')

    for i in (bar := trange(num_batches)):
        if eval_every > 0 and i % eval_every == 0 and not skip_eval:

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

        opt.zero_grad()
        if warmup > 0:
              sch.step()

        # sample a batch from the data
        source, target = sample_batch(data, context, model_batch_size)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        output = model(source)
        loss = F.cross_entropy(output.transpose(2, 1), target)

        loss.backward()
        opt.step()

        wandb.log({
            'loss': loss,
            'learning_rate': sch.get_last_lr()[0],
            'gradient_norm': gradient_norm(model),
            'pre-training': 0.0
        })

        bar.set_postfix({'loss': f'{loss:.02}'})

if __name__ == '__main__':
    fire.Fire(go)