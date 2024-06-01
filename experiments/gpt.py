
import up
from up.util import tic, toc, coords, d, sample, sample_sequence, gradient_norm
from up.data import load_data, cas

from former.util import d, here, tic, toc, sample_batch, enwik8_string, enwik8_bytes, estimate_compression
import former

import wandb, random, fire, gzip

import torch
from torch import nn
import torch.nn.functional as F

from math import floor, sqrt

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

    # print('---')
    # for seq in batch:
    #     seq = bytes(list(seq)).decode('utf-8')
    #     print(seq)
    #     print()
    # print('---')


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
         sequential=False, eval_samples=10_000, mlm_prob=0.15, ascii_only=False,
         init_mult_max=5.0, mask_prob_max=0.7, nonlinearity='relu', dropout=0.1,
         skip_eval=False,         # Whether to skip the evaluation
         eval_ood=False,          # Whether to evaluate on OOD datasets
         name=None,               # WandB name
         eval_batch_mult=2.0,     # How much bigger the eval batches can be than the training batches
         pre_file=None,           # File containing pre-training data
         accumulate = 1,          # The number of batches to accumulate the gradient over before a gradient step occurs
         model_file = None,            # Filename of a pretrained model/optimizer
         model_dst = './pretrained-{}.pt', # Where to save the pretrained model Add in an {} for the number of instances
         cp_every = 100_000,       # Save a checkpoint for the model every n batches.
         dp = False,                # Use data-parallel
         wandb_project = 'up',
         eval_at = (60_000, 120_000), # Evaluate at these points during finetuning
         scalefactor=None,
         acc_warmup=-1
       ):

    """
    Generates a dataset by sampling sequences autoregressively from a given model.

    Separates model and source. Samples over all depths of universal model.
    """

    wd = wandb.init(
        name=name,
        project=wandb_project,
        tags=tags,
        config=locals(),
        mode= 'disabled' if debug else 'online'
    )

    if eval_ood:
        datasets = {
            'champ': torch.tensor(load_data('champ'), dtype=torch.long),
            'dyck' : torch.tensor(load_data('dyck'), dtype=torch.long),
            'ndfa' : torch.tensor(load_data('ndfa'), dtype=torch.long),
            'toy'  : torch.tensor(load_data('toy'),  dtype=torch.long),
            'bits' : torch.tensor(load_data('bits'), dtype=torch.long),
            'wp'   : torch.tensor(load_data('wp-val'), dtype=torch.long),
        }
    else:
        datasets = {
            'wp'   : torch.tensor(load_data('wp-val'), dtype=torch.long)
        }

    scaler = torch.cuda.amp.GradScaler()

    # Target for training
    model = up.GTransformer(emb=emb, heads=heads, depth=mdepth, seq_length=context, num_tokens=NUM_TOKENS,
                            scalefactor=scalefactor, dropout=dropout)

    if torch.cuda.is_available():
        model.cuda()

    if dp:
        model = torch.nn.DataParallel(model)

    # Throughput test to find batch size
    if model_batch_size is None:
        dummy_input = torch.randint(low=0, high=NUM_TOKENS, size=(1, context), dtype=torch.long, device=d())

        def dummy_loss(output):
            b, c, e = output.size()
            dummy_target = torch.randint(low=0, high=NUM_TOKENS, size=(b, c), dtype=torch.long, device=d())
            return F.cross_entropy(output.transpose(1, 2), dummy_target)

        print('Starting throughput test.');
        tic()
        model_batch_size, batch_sizes, throughputs = up.util.find_batch_size(model=model, loss=dummy_loss,
                                                                             input=dummy_input, burn_in=3, samples=20,
                                                                             wandb=None, use_amp=True)

        print(f'Finished ({toc():.4}s). Best batch size found: {model_batch_size}. Batch sizes and throughputs: {zip(batch_sizes, throughputs)}.')

    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    if warmup > 0:
        max_lr = lr
        lrdelta = lr / warmup
        set_lr(0.0, opt)

    if acc_warmup > 0:
        max_acc = accumulate
        accdelta = accumulate / acc_warmup
        accumulate_current = 1.0

    # Load a pretrained model
    if model_file is not None:
        checkpoint = torch.load(model_file)

        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    # Pretrain a model
    else:

        if pre_file is None:
            cmp_source = \
                up.ConditionalTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=NUM_TOKENS) \
                if sequential else \
                up.GTransformer(emb=emb, heads=heads, depth=cdepth, seq_length=context, num_tokens=NUM_TOKENS, nl=nl(nonlinearity), mask_channel=True)

            if torch.cuda.is_available():
                cmp_source.cuda()

            if dp:
                cmp_source = torch.nn.DataParallel(cmp_source)

            buffer = torch.randint(low=0, high=NUM_TOKENS, size=(buffer_size, context), device=d())

        else:
            with gzip.open(pre_file, 'r') as file:
                buffer = file.read()
                buffer = torch.tensor([int(byte) for byte in buffer], dtype=torch.long)
                buffer = buffer.reshape(-1, context)

                buffer_size = buffer.size(0)

        sampletime = -1.0

        # Pretraining batches
        if pre_batches > 0:

            print('Start pre-training')
            gnm, gnv = 0, 0  # moving avg and std of the gradient norm

            for i in (bar := trange(pre_batches)):

                if cp_every > 0 and i > 0 and i % cp_every == 0:

                    if model_dst is not None:
                        print(f'Saving model at {i * model_batch_size}.')
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                        }, model_dst.format(i * model_batch_size))

                if eval_every > 0 and i % eval_every == 0 and not skip_eval:

                    for name, data in datasets.items():
                        print(f'evaluating {name}')

                        with torch.no_grad():
                            est = estimate_compression(
                                model=model,
                                data=data,
                                nsamples=eval_samples,
                                context=context,
                                batch_size=int(model_batch_size * eval_batch_mult),
                                model_produces_logits=True
                            )

                        wandb.log({f'val-{name}': est})

                if pre_file is None: # Sample on the fly.

                    tic()
                    with torch.no_grad():
                        # Re-initialize the parameters of source (i.e. sample a random source)
                        up.weights_init_minimal(cmp_source, init_mult_max=init_mult_max)

                        # slice a random selection of rows from the buffer (without replacement)
                        iz = random.sample(range(buffer.size(0)), sample_batch_size)
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

                            chars, mask = output[:, :, :-1], output[:, :, -1]

                            chars = sample(chars, temperature=temperature)
                            mask = torch.sigmoid(mask).to(torch.bool)

                            z[mask] = chars[mask]

                            buffer[iz, :] = z

                        # -- The output of sample_sequence is context + 1 because of the seed, so we slice off the last character. The
                        #    seed is likely more important in the long run

                        # -- Note that the samples are in full precision. These often require large weights, so mixed precision
                        #    leads to nans and infs and whatnot.

                    sampletime = toc()

                tic()
                # Perform a training step on batches sampled from the buffer

                iz = random.sample(range(buffer_size), model_batch_size)

                batch = buffer[iz, :]
                if torch.cuda.is_available():
                    batch = batch.cuda()

                input  = batch[:, :-1]
                target = batch[:, 1:]

                with torch.cuda.amp.autocast():
                    output = model(input)
                    loss = F.cross_entropy(output.transpose(2, 1), target)

                scaler.scale(loss).backward()

                if i % int(floor(accumulate_current)) == 0: # perform a step

                    gn = gradient_norm(model)
                    lim = gnm + sqrt(gnv) * gc
                    if i > 100 and gn > lim:
                        nn.utils.clip_grad_norm_(model.parameters(), lim)

                    gnm, gnv = up.util.em_meanvar(gn, gnm, gnv)
                    for parm in model.parameters():
                        parm.grad *= 1.0 / floor(accumulate_current)

                    scaler.step(opt)
                    scaler.update()

                    opt.zero_grad()

                    wandb.log({
                        'gradient_norm': gn,
                    })

                    traintime = toc()

                wandb.log({
                    'loss': loss,
                    'learning_rate': lr,
                    'sample_time': sampletime,
                    'train_time': traintime,
                    'pre-training': 1.0,
                    'acc': accumulate_current,
                    'ema_gn': gnm,
                    'em_std_gn': sqrt(gnv),
                    'clip': 1.0 if (gn > lim) and (i > 100) else 0.0
                })
                bar.set_postfix({'loss': f'{loss:.02}'})

                if i % print_every == 0:

                    print('target')
                    print_batch(batch[:4, :], ascii_only)

                    print('model output')

                    seed = torch.randint(low=0, high=NUM_TOKENS, size=(4, 1), device=d())
                    output = sample_sequence(model, seed, context, num_tokens = NUM_TOKENS, length=context,
                                                    temperature = temperature)
                    print_batch(output, ascii_only)

                if lr < max_lr:
                    lr += lrdelta
                    set_lr(lr, opt=opt)

                if accumulate_current < max_acc:
                    accumulate_current += accdelta

    # Fine-tuning
    print('Start finetuning')


    if warmup > 0:
        set_lr(0.0, opt)

    # -- We keep the same optimizer (the statistics may well carry over from pre-training to finetuning)
    #    but we reset the learning rate warmup.

    if acc_warmup > 0:
        accumulate_current = 1.0

    traindata = torch.tensor(up.data.load_data('wp-train'), device='cpu')
    gnm, gnv = 0, 0 # moving avg and std of the gradient norm

    for i in (bar := trange(num_batches)):
        if (eval_every > 0 and i % eval_every == 0 and not skip_eval) or (i in eval_at):

            for name, valdata in datasets.items():
                print(f'evaluating {name}')

                with torch.no_grad():
                    est = estimate_compression(
                        model=model,
                        data=valdata,
                        nsamples=eval_samples,
                        context=context,
                        batch_size=int(model_batch_size * eval_batch_mult),
                        model_produces_logits=True
                    )

                wandb.log({f'val-{name}': est})

        # sample a batch from the data
        source, target = sample_batch(traindata, context, model_batch_size)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        with torch.cuda.amp.autocast():
            output = model(source)
            loss = F.cross_entropy(output.transpose(2, 1), target)

        scaler.scale(loss).backward()

        if i % int(floor(accumulate_current)) == 0:  # perform a step

            # Adaptive gradient clipping. We keep an exponential moving estimate of the mean and variance of the gradient
            # norm, and if the current norm is more than `cfg.up.gc` standard deviations above the mean, we clip it to
            # that value.
            gn = gradient_norm(model)
            lim = gnm + sqrt(gnv) * gc
            if i > 100 and gn > lim:
                nn.utils.clip_grad_norm_(model.parameters(), lim)

            gnm, gnv = up.util.em_meanvar(gn, gnm, gnv)
            for parm in model.parameters():
                parm.grad *= 1.0 / floor(accumulate_current)

            scaler.step(opt)
            scaler.update()

            opt.zero_grad()

            wandb.log({
                'gradient_norm': gn,
            })

            traintime = toc()

        wandb.log({
            'loss': loss,
            'learning_rate': lr,
            'acc': accumulate_current,
            'pre-training': 0.0,
            'ema_gn': gnm,
            'em_std_gn': sqrt(gnv),
            'clip': 1.0 if (gn > lim) and (i > 100) else 0.0
        })

        bar.set_postfix({'loss': f'{loss:.02}'})

        if lr < max_lr:
            lr += lrdelta
            set_lr(lr, opt=opt)

        if accumulate_current < max_acc:
            accumulate_current += accdelta

def set_lr(lr, opt):
    for g in opt.param_groups:
        g['lr'] = lr
        g['initial_lr'] = lr

def grad_scale(model, scale):
    for parm in model.parameters():
        parm.grad *= scale

if __name__ == '__main__':
    fire.Fire(go)