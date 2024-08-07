import up
from up.util import tic, toc, coords, d, sample, sample_sequence, gradient_norm
from up.data import load_data, cas

from former.util import d, here, tic, toc, sample_batch, enwik8_string, enwik8_bytes, estimate_compression
import former

import wandb, random, fire, gzip, math, tqdm, os, json

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

    if name == 'gelu':
        return torch.nn.functional.gelu

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

    return int(round( (math.log(width) - a) / b ))

def go(
         width : int,               # Scaling step for the width (steps of 64)
         width_per_step=128,
         min_heads=32,                # minimum nr of heads
         width_per_head=128,          # dimension per head (above the minimum)
         context=512,
         temperature=0.5,
         target_microbatch_size=52,   # microbatch size for the target model
         macrobatch_size=256,         # macrobatch size for the target model
         source_batch_mult=2.0,       # how much bigger the batch size is for the source model than the microbatch size
                                      # for the target model
         buffer_size_mult=20,         # How much bigger the buffer is than one source batch
         batches=300_000,             # For how many microbatches to pretrain
         reset_prob=0.01,
         width0 = 640,                # The width for which the base learning rate is tuned
         base_lr=3e-4,                # Base learning rate at width0
         debug=False,
         warmup=100_000,
         eval_every=500_000,            # How often (in microbatches) to evaluate
         print_every=500,             # How often to print the source output
         gc=1.0,                      # Gradient clipping.
         eval_samples=10_000,         # On how many samples to evaluate
         weight_mult1=1.4,            # multiplier for the weights of the source model
         weight_mult2=10,             # multiplier for the weights (grp 1) of the source model
         weight_multb=1,              # multiplier for the weights (grp 2) of the source model
         source_mask=False,           # Whether to apply masking to the source model
         mask_prob_max=0.7,
         skip_eval=False,             # Whether to skip the evaluation
         eval_ood=True,               # Whether to evaluate on OOD datasets
         name=None,                   # WandB name
         eval_batch_mult=2.0,         # How much bigger the eval batches can be than the training batches
         cp_every = 500_000,          # Save a checkpoint for the model every n instances.
         dp = False,                  # Use data-parallel (multi GPU)
         mbwarmup = 100_000,          # Accumulation warmup (in instances)
         mb_min = 16,                 # Minimum microbatch size to start warming up from.
         mb_start = 100_000,          # Number of instances to wait before starting the warmup
         old_init=False,
         init_factor=1,               # Multiplier for the muP init
         skip_mup=False,
         out_factor=1,                # Logit multiplier in the model
         weight_decay=0.0,            # weight decay
         sqrt_attn_scale=False,       # Use the original sqrt attention scaling
         source_width=None,           # Width factor of the source model (if None, the same as the target)
         source_microbatch_size=None,
         nl_source='relu',
         nl_target='relu',
         kqnorm=False,
         save_to=None,
         load_teacher=None,
         teacher_alpha=0.5,
         twidth=384,
         tout_factor=32
):

    """
    Scaling experiment. Uses Levine 2021 to get the depth/width and Yang 2022 to scale the lr and initialization.

    We first scale the width (d_model) by the number of heads. Note that the nr of heads itself doesn't matter very much,
    but it's a convenient way to scale the width. We hold to the convention that the number of dimensions in the width
    per head should be constant (at 128 in our case).

    We then use the principles from Yang 2022 (muP parametrization) to scale the weight initialization and the learning
    rate.

    The context length and (macro)batch size are fixed. We keep the source model and target model the same size and use
    the same intialization for both, except that the initialization of the source model is multiplied by `init_mult_max`.

    """

    # Compute some values that should be logger to wandb
    # width = wfactor * width_per_step
    depth = get_depth(width)

    if source_microbatch_size is None:
        source_microbatch_size = int(round(target_microbatch_size * source_batch_mult))
    buffer_size = int(round(source_microbatch_size * buffer_size_mult))

    heads = max(width//width_per_head, min_heads)
    assert width % heads == 0

    swidth = width if source_width is None else width_per_step * source_width
    sdepth =get_depth(swidth)
    sheads = max(swidth//width_per_head, min_heads)
    assert width % heads == 0

    wd = wandb.init(
        name=name,
        project='up-scaling',
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
    model = up.GTransformer(emb=width, heads=heads, depth=depth, seq_length=context, nl=nl(nl_target),
                            num_tokens=NUM_TOKENS, nosqrt=not sqrt_attn_scale, output_mult=out_factor, kqnorm=kqnorm)

    if torch.cuda.is_available():
        model.cuda()
    if dp:
        model = torch.nn.DataParallel(model)

    if not skip_mup:
        opt = model.mup(base_lr=base_lr, width0=width0, factor=init_factor, optcls=torch.optim.AdamW, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(lr=base_lr, params=model.parameters(), weight_decay=weight_decay)

    if warmup > 0:
        # warmup = warmup / accumulate
        # sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (warmup / model_batch_size), 1.0))
        for g in opt.param_groups:
            g['max_lr'] = g['lr']
            g['lr_delta'] = g['lr'] / warmup

            g['lr'] = 0.0

    print(opt)

    cmp_source = up.GTransformer(emb=swidth, heads=sheads, depth=sdepth, seq_length=context, num_tokens=NUM_TOKENS, nl=nl(nl_source), mask_channel=True)

    if torch.cuda.is_available():
        cmp_source.cuda()

    if dp:
        cmp_source = torch.nn.DataParallel(cmp_source)

    # Load teacher model (if specified)
    teacher = None
    if load_teacher:
        loaded = torch.load(load_teacher, map_location=d())

        tdepth = get_depth(twidth)
        theads = max(twidth//width_per_head, min_heads)
        tnl = nl_target
        tcontext = context

        teacher = up.GTransformer(emb=twidth, heads=theads, depth=tdepth, seq_length=tcontext, num_tokens=NUM_TOKENS,
                                     nl=nl(tnl), nosqrt=not sqrt_attn_scale, output_mult=tout_factor, kqnorm=False,
                                     mask_channel=False)
        teacher.load_state_dict(loaded['model_state_dict'])
        print('Teacher loaded.')

    buffer = torch.randint(low=0, high=NUM_TOKENS, size=(buffer_size, context), device=d())

    sampletime = -1.0

    print('Start pre-training')

    accumulated = 0 # nr of instances accumulated currently
    if mbwarmup > 0:
        mbraw = mb_min # macrobatch size as a float
        mbdelta = (macrobatch_size - mb_min) / mbwarmup
    else:
        mbraw = macrobatch_size

    instances_seen = 0
    last_eval = float('-inf')

    for i in (bar := trange(batches)):

        if cp_every > 0 and i > 0 and i % cp_every == 0:

            if save_to is not None:
                print(f'Saving model at {i} instances.')
                # torch.save({
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': opt.state_dict(),
                # }, save_to.format(i))

                torch.save(model, save_to.format(i))
                # Save just the model. This is a bit brittle to code changes, but doesn't require us to save the
                # hyperparams manually

        if instances_seen - last_eval > eval_every and not skip_eval:

            for name, data in datasets.items():
                print(f'evaluating {name}')

                with torch.no_grad():
                    est = estimate_compression(
                        model=model,
                        data=data,
                        nsamples=eval_samples,
                        context=context,
                        batch_size=int(target_microbatch_size * eval_batch_mult),
                        model_produces_logits=True
                    )

                wandb.log({f'val/{name}': est}, step=instances_seen)

            last_eval = instances_seen

        # Sample noise from a random model and insert into the buffer
        tic()
        with torch.no_grad():
            # Re-initialize the source

            if old_init:
                up.weights_init(cmp_source, init_mult_max=50.0, mask_prob_max=0.7)
            else:
                up.weights_init_mup(cmp_source, mult1=weight_mult1, mult2=weight_mult2, multb=weight_multb, mask=source_mask)

            # slice a random selection of rows from the buffer (without replacement)
            iz = random.sample(range(buffer.size(0)), source_microbatch_size)
            z = buffer[iz, :]

            # replace some random rows with uniform random characters
            rows = torch.bernoulli(torch.full(size=(source_microbatch_size, 1), fill_value=reset_prob))
            mask = rows.expand(source_microbatch_size, context).to(torch.bool)

            uniform = torch.randint(low=0, high=NUM_TOKENS, size=(source_microbatch_size, context), device=d())
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

        bs = min(int(round(mbraw)), target_microbatch_size)
        # If the current macrobatch sizer is smaller than the microbatch size, we go with the smaller value
        # (i.e. we leave memory empty).

        iz = random.sample(range(buffer_size), bs)

        batch = buffer[iz, :]
        if torch.cuda.is_available():
            batch = batch.cuda()

        input  = batch[:, :-1]
        target = batch[:, 1:]

        with torch.cuda.amp.autocast():
            output = model(input)
            rloss = F.cross_entropy(output.transpose(2, 1), target, reduction='sum')

            if teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(model)

                tloss = F.cross_entropy(output.transpose(2, 1), teacher_out.softmax().transpose(2, 1), reduction='sum')
            else:
                tloss = 0.0

        loss = (rloss / input.size(1)) + teacher_alpha * (tloss / input.size(1))
        # divide out the time, but sum over the instances

        scaler.scale(loss).backward()
        accumulated += input.size(0)

        if accumulated >= mbraw: # perform a step

            scaler.unscale_(opt)

            # scale the gradients to average over the macrobatch
            # -- here we divide out the instances
            for parm in model.parameters():
                parm.grad /= accumulated

            gn = gradient_norm(model)
            if gc > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gc)

            wandb.log({
                'gradient_norm': gn,
                'accumulated': accumulated # Sanity check.
            }, step=instances_seen)

            scaler.step(opt)
            scaler.update()

            opt.zero_grad()

            accumulated = 0
            acc_last = i

        traintime = toc()

        wandb.log({
            'loss': loss.item() / input.size(0),
            'sample_time': sampletime,
            'train_time': traintime,
            'pre-training': 1.0,
            'mbraw': mbraw
        }, step=instances_seen)

        if skip_mup:
            wandb.log({
                'learning_rate (stable)': opt.param_groups[0]['lr'],
            }, step=instances_seen)
        else:
            wandb.log({
                'learning_rate (stable)': opt.param_groups[0]['lr'],
                'learning_rate (scaled)': opt.param_groups[1]['lr'],
            }, step=instances_seen)

        wandb.log({}, step=instances_seen, commit=True)

        bar.set_postfix({'loss': f'{rloss.item():.02}'})

        if mbwarmup > 0 and mbraw < macrobatch_size and instances_seen > mb_start:
            mbraw += mbdelta * batch.size(0)

        if warmup > 0:
            for g in opt.param_groups:
                if g['lr'] < g['max_lr']:
                    g['lr'] += g['lr_delta'] * batch.size(0)

        if i % print_every == 0:

            print('target samples', i)
            print_batch(batch[:4, :], False)

        instances_seen += batch.size(0)

def throughput(fr=3, to=12, context=512, samples=40, burn_in=10, width_per_head=128):
    """
    :return:
    """

    res = {}
    for heads in trange(fr, to + 1):

        # Compute some values that should be logger to wandb
        width = heads * width_per_head
        depth = get_depth(width)

        print('depth:', depth, ', width: ', width)

        # Target for training
        model = up.GTransformer(emb=width, heads=heads, depth=depth, seq_length=context, num_tokens=NUM_TOKENS,
                                nosqrt=True)
        if torch.cuda.is_available():
            model.cuda()

        dummy_input = torch.randint(low=0, high=NUM_TOKENS, size=(1, context), dtype=torch.long, device=d())

        def dummy_loss(output):
            b, c, e = output.size()
            dummy_target = torch.randint(low=0, high=NUM_TOKENS, size=(b, c), dtype=torch.long, device=d())
            return F.cross_entropy(output.transpose(1, 2), dummy_target)

        print('Starting throughput test.');
        tic()
        model_batch_size, batch_sizes, throughputs = up.util.find_batch_size(model=model, loss=dummy_loss,
                                                                             input=dummy_input, burn_in=burn_in,
                                                                             samples=samples, wandb=None, use_amp=True)

        res[heads] = {
            'best' :  model_batch_size,
            'all' : list(zip(batch_sizes, throughputs))
        }

    print('Finished. Best results:')
    for heads in range(fr, to + 1):
        print(heads, '\t', res[heads]['best'])

    print('\n\n\n')
    print('All results.')
    print(res)

    with open('throughput.json', 'w') as file:
        json.dump(res, file, indent=6)


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