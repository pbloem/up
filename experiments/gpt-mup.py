import up
from up.util import tic, toc, coords, d, sample, sample_sequence, gradient_norm
from up.data import load_data, cas

from former.util import d, here, tic, toc, sample_batch, enwik8_string, enwik8_bytes, estimate_compression
import former

import wandb, random, fire, gzip, math, tqdm, os

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

def get_depth(width):
    """
    Compute the optimal depth for a transformer model of a given width.

    :param width: 
    :return: 
    """
    # Constants from the paper
    a, b = 5.039, 5.55e-2

    return int(round(math.log(width) - a) / b)

WIDTHPERHEAD = 128

def go(
         heads : int,                 # Number of heads. The rest of the model parameters are derived from this.
         context=512,
         temperature=0.5,
         target_microbatch_size=100,  # microbatch size for the target model
         source_batch_mult=2.0,       # how much bigger the batch size is for the source model than the microbatch size
                                      # for the target model
         buffer_size_mult=20,         # How much bigger the buffer is than one source batch
         batches=300_000,             # For how many microbatches to pretrain
         reset_prob=0.01,
         heads0 = 5,                  # The number of heads for which the base learning rate is set
         base_lr=3e-4,                # Base learning rate at heads0
         debug=False,
         warmup=100_000,
         eval_every=5_000,            # How often (in microbatches) to evaluate
         print_every=500,             # How often to print the source output
         gc=1.0,                      # Gradient clipping.
         eval_samples=10_000,         # On how many samples to evaluate
         mlm_prob=0.15,
         init_mult_max=50.0,          # multiplier for the weights of the source model
         mask_prob_max=0.7,
         nonlinearity='relu',
         skip_eval=False,             # Whether to skip the evaluation
         eval_ood=False,              # Whether to evaluate on OOD datasets
         name=None,                   # WandB name
         eval_batch_mult=2.0,         # How much bigger the eval batches can be than the training batches
         accumulate = 1,              # The number of batches to accumulate the gradient over before a gradient step occurs
         cp_every = 100_000,          # Save a checkpoint for the model every n batches.
         dp = False,                  # Use data-parallel (multi GPU)
         acc_warmup = 0               # Accumulation warmup (in instances)
       ):

    """
    Scaling experiment. Uses Levine 2021 to det the depth/width and Yang 2022

    We first scale the width (d_model) by the number of heads. Note that the nr of heads itself doesn't matter very much,
    but it's a convenient way to scale the width. We hold to the convention that the number of dimensions in the width
    per head should be constant (at 110 in our case).

    We then use the principles from Yang 2022 (muP parametrization) to scale the weight initialization and the learning
    rate.

    The context length and (macro)batch size are fixed. We keep the source model and target model the same size and use
    the same intialization for both, except that the initialization of the source model is multiplied by `init_mult_max`.

    """

    width = heads * WIDTHPERHEAD
    depth = get_depth(width)

    wd = wandb.init(
        name=name,
        project='up',
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

    print('depth:', depth, ', width: ', width)

    # Target for training
    model = up.GTransformer(emb=width, heads=heads, depth=depth, seq_length=context, num_tokens=NUM_TOKENS, nosqrt=True)


    if torch.cuda.is_available():
        model.cuda()
    if dp:
        model = torch.nn.DataParallel(model)

    opt = model.mup(base_lr=base_lr, width0=heads0 * WIDTHPERHEAD)

    print(opt)

    exit()

    if warmup > 0:
        # warmup = warmup / accumulate
        # sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / model_batch_size), 1.0))

        maxlr = lr
        lr = 0.0
        lrdelta = maxlr / warmup

    else:
        if mult_lr:
            lr = lr * accumulate

    # Load a pretrained model
    if model_file is not None:
        checkpoint = torch.load(model_file)

        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

    # Pretrain a model
    else:

        if pre_file is None:
            cmp_source = up.GTransformer(emb=cemb, heads=cheads, depth=cdepth, seq_length=context, num_tokens=NUM_TOKENS, nl=nl(nonlinearity), mask_channel=True)

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

            accumulated = acc_last = 0
            if acc_warmup > 0:
                accraw = 1.0
                accdelta = (accumulate - 1) / acc_warmup
            else:
                accraw = accumulate
                accdelta = 0.0

            for i in (bar := trange(pre_batches)):

                # if cp_every > 0 and i > 0 and i % cp_every == 0:
                #
                #     if model_dst is not None:
                #         print(f'Saving model at {i * model_batch_size}.')
                #         torch.save({
                #             'model_state_dict': model.state_dict(),
                #             'optimizer_state_dict': opt.state_dict(),
                #         }, model_dst.format(i * model_batch_size))

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
                        up.weights_init(cmp_source, init_mult_max=init_mult_max, mask_prob_max=mask_prob_max)

                        # slice a random selection of rows from the buffer (without replacement)
                        iz = random.sample(range(buffer.size(0)), sample_batch_size)
                        z = buffer[iz, :]

                        # replace some random rows with uniform random characters
                        rows = torch.bernoulli(torch.full(size=(sample_batch_size, 1), fill_value=reset_prob))
                        mask = rows.expand(sample_batch_size, context).to(torch.bool)

                        uniform = torch.randint(low=0, high=NUM_TOKENS, size=(sample_batch_size, context), device=d())
                        z[mask] = uniform[mask]

                        # pass it through a randomly chosen model
                        output = cmp_source(z)

                        # The model generates an output sequence as well as a mask. The final sequence has the output
                        # characters at the masked positions and the input characters at the non-masked positions. The
                        # idea is that this makes it more likely for the model to retain any structure present in the
                        # input.
                        # -- It is theoretically possible for the model to do this without the masking, but it's very
                        #    unlikely with random parameters.
                        chars, mask = output[:, :, :-1], output[:, :, -1]

                        chars = sample(chars, temperature=temperature)
                        mask = torch.sigmoid(mask).to(torch.bool)

                        z[mask] = chars[mask]

                        buffer[iz, :] = z

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
                    rloss = F.cross_entropy(output.transpose(2, 1), target)
                    loss = rloss / accumulate # scale the loss to compensate for the accumulation

                scaler.scale(loss).backward()
                accumulated += 1

                acc_current = min(int(round(accraw)), accumulate)
                if i > acc_last + acc_current: # perform a step

                    gn = gradient_norm(model)

                    if gc > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), gc)

                    wandb.log({
                        'gradient_norm': gn,
                        'accumulated': accumulated # Sanity check.
                    })

                    scaler.step(opt)
                    scaler.update()

                    opt.zero_grad()

                    accumulated = 0
                    acc_last = i

                traintime = toc()

                wandb.log({
                    'loss': rloss.item(),
                    'learning_rate': lr * acc_current if mult_lr else lr,
                    'sample_time': sampletime,
                    'train_time': traintime,
                    'pre-training': 1.0,
                    'accumulate': accraw
                })
                bar.set_postfix({'loss': f'{rloss.item():.02}'})

                if acc_warmup > 0 and accraw < accumulate:
                    accraw += accdelta * batch.size(0)

                if warmup > 0:
                    lr += lrdelta * batch.size(0)

                    set_lr(lr * acc_current if mult_lr else lr, opt)

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

    traindata = torch.tensor(up.data.load_data('wp-train'), device='cpu')

    for i in (bar := trange(num_batches)):
        if eval_every > 0 and i % eval_every == 0 and not skip_eval:

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

        opt.zero_grad()

        # sample a batch from the data
        source, target = sample_batch(traindata, context, model_batch_size)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        with torch.cuda.amp.autocast():
            output = model(source)
            loss = F.cross_entropy(output.transpose(2, 1), target)

        scaler.scale(loss).backward()

        gn = gradient_norm(model)
        if gc > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), gc)

        scaler.step(opt)
        scaler.update()

        if warmup > 0:
              sch.step()

        wandb.log({
            'loss': loss,
            'learning_rate': sch.get_last_lr()[0],
            'gradient_norm': gn,
            'pre-training': 0.0
        })

        bar.set_postfix({'loss': f'{loss:.02}'})

def throughput():
    """
    TODO: Run throughput tests for a range of model sizes.
    :return:
    """

    pass

def coord_check(depth=12, steps=3, context=512, model_batch_size=32, disable_mup=False, max_width=14, nocuda=False):
    """
    Sanity check for the muP implementation. The output activations at each layer should have the same magnitude,
    regardless of width.

    :return:
    """

    if nocuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ","

    import matplotlib as mpl
    mpl.use('Agg')
    from matplotlib import pyplot as plt

    widths = [2 ** e for e in range(7, max_width+1)]

    res = {d:[] for d in range(depth + 3)}

    model = None
    for width in tqdm.tqdm(widths):

        del model
        model = up.GTransformer(emb=width, heads=width // 32, depth=depth, seq_length=context, num_tokens=NUM_TOKENS,
                                nosqrt=True)

        if torch.cuda.is_available() and not nocuda:
            model.cuda()

        if disable_mup:
            opt = torch.optim.Adam(lr=3e-4, params=model.parameters())
        else:
            opt = model.mup(base_lr=3e-4, width0=640)

        # Train for a small number of batches on WP
        traindata = torch.tensor(up.data.load_data('wp-train'), device='cpu')
        for i in range(steps):

            opt.zero_grad()
            source, target = sample_batch(traindata, context, model_batch_size)

            if torch.cuda.is_available() and not nocuda:
                source, target = source.cuda(), target.cuda()

            output = model(source)
            loss = F.cross_entropy(output.transpose(2, 1), target)

            loss.backward()
            opt.step()

        with torch.no_grad():

            dkey = 0
            x, _ = sample_batch(traindata, context, model_batch_size)
            if torch.cuda.is_available() and not nocuda:
                x = x.cuda()

            tokens = model.token_embedding(x)
            b, t, e = tokens.size()

            positions = model.pos_embedding(torch.arange(t, device='cpu' if nocuda else d()))[None, :, :].expand(b, t, e)
            x = tokens + positions

            l1 = x.abs().mean().item()
            res[dkey].append((width, l1))
            dkey += 1

            for i, block in enumerate(model.tblocks):
                x = block(x)

                l1 = x.abs().mean().item()
                res[dkey].append((width, l1))
                dkey += 1

            x = model.toprobs(x)

            l1 = x.abs().mean().item()
            res[dkey].append((width, l1))


    plt.figure()
    for dkey in res.keys():
        widths = [w for w, _ in res[dkey]]
        l1s = [l for _, l in res[dkey]]
        plt.plot(widths, l1s, label=str(dkey))

    plt.xlabel('width')
    plt.ylabel('l1')
    plt.legend()
    plt.title(f'Coord check with: {depth=}, {steps=}, {context=} and muP {"disabled" if disable_mup else "enabled"}')

    plt.savefig(f'coordcheck.d{depth}.mup{not disable_mup}.pdf')


def set_lr(lr, opt):
    for g in opt.param_groups:
        g['lr'] = lr
        g['initial_lr'] = lr


if __name__ == '__main__':
    fire.Fire()