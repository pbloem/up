import up
from up.util import tic, toc, coords, d, sample, sample_sequence, gradient_norm, remap
from up.data import load_data, cas, gen_autseq
from up import ProgTransformerBlock

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

def get_flops(model, batch_size, ctx, backward=False):
    """
    Estimates the number of flops spent in one forward and one backward at the given microbatch size.
    :param model:
    :param batch_size:
    :param ctx:
    :return:
    """

    scaler = torch.cuda.amp.GradScaler()

    input  = torch.randint(low=0, high=NUM_TOKENS, size=(batch_size, ctx), device=d())

    if backward:
        target = torch.randint(low=0, high=NUM_TOKENS, size=(batch_size, ctx), device=d())
        opt = torch.optim.AdamW(lr=3e-4, params=model.parameters())

    with torch.profiler.profile(
            with_flops=True,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
    ) as p:
        with torch.cuda.amp.autocast():

            output = model(input)

            if backward:
                loss = F.cross_entropy(output.transpose(2,1), target)
                scaler.scale(loss).backward()
                scaler.step(opt)
                opt.zero_grad()

    # total_gflops = sum(k.flops for k in p.key_averages()) / 1e9
    total_flops = sum([e.flops for e in p.events()])
    # -- profiler is very poorly documented... Snippet
    #    from https://github.com/pytorch/pytorch/issues/69782

    return total_flops

def rsamp(x):
    if type(x) in (tuple, list):
        mi, ma = min(x), max(x)
        rng = ma - mi
        return random.random() * rng + mi

    return x

def dsamp(x):
    if type(x) in (tuple, list):
        return random.choice(x)

    return x

def antisol_batch(model, batch, numchars, num=5, seed_length=5, context=64, verbose=False, use_mask=False):
    """
    Replaces `num` instances in the batch by by anti-Solomonoff instances generated from `model`
    :param model:
    :param batch:
    :param num:
    :param seed_length:
    :param context: Max context to use. Shortening this can speed up generation, but worsens the quality of the
        generated string.
    :param mask: Limit the sampled characters to those in the seed and mask the rest.
    :return:
    """
    b, l = batch.size()

    assert b >= num
    idxs = random.sample(population=range(b), k=num)
    seeds = batch[idxs, :seed_length]

    if use_mask:
        wlist = [set(chars.tolist()) for chars in seeds]
        mask = [[c for c in range(numchars) if c not in wl] for wl in wlist]
        # -- All the characters not in the seed for each instance in the batch
        # -- I can't think of a parallelized way to implement the masking. Here's hoping the impact is minimal.
        #    This might be a bit expensive for large token vocabs.
        # -- If the seed contains only one character, the sampled sequence is necessarily that same character repeated.
        #    For now, that doesn't seem too bad (it's a worthwhile pattern to learn from, and it should be rare).

    with torch.no_grad():
        as_strings = antisol(model, seeds, length=l, context=context, verbose=verbose, mask=mask if use_mask else None)

    batch[idxs, :] = as_strings

def antisol(model, seeds, length, context, verbose=False, mask=None):
    """
    Samples anti-Solomonoff strings from the model for the given batch of seeds

    :param model:
    :param seeds:
    :param length:
    :return:
    """

    b, sl = seeds.size()
    sequence = seeds.detach().clone()

    for _ in range(length - sl):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[:, -context:]

        # Run the current input through the model
        output = model(input)

        if mask is None:
            # Take the least probable token as a "sample"
            samples = output[:, -1, :].argmin(dim=-1) # Note the arg_min_
            assert samples.size() == (b, )
        else:
            samples = []
            output = output.detach().clone()
            for inst, imask in zip(output[:, -1, :], mask):
                inst[imask] = float('inf')
                samples.append(inst.argmin().item())
            samples = torch.tensor(samples, device=seeds.device)

        sequence = torch.cat( (sequence, samples[:, None]), dim=1) # Append the sampled token to the sequence

    if verbose:
        print('anti-Solomonoff strings')
        for inst in sequence:
            inst = remap(inst.tolist())
            print(''.join(str(c) for c in inst))
        print('\n\n')

    return sequence

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
         cooldown=-1,                 # After the warmup finishes, we cool down by halving the lr every `cooldown` instances
         eval_every=500_000,          # How often (in microbatches) to evaluate
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
         attn_factor=1.0,             # additional scaling factor for the attention weight (pre-softmax)
         source_width=None,           # Width factor of the source model (if None, the same as the target)
         source_microbatch_size=None,
         source='transformer',        # Source data generator (transformer, uniform, ndfa, pointwise)
         source_order=3,              # Order for the Markov data generator
         source_alpha=0.5,            # alpha for the pointwise generator
         sequential=False,            # Whether to use a sequential sampling model (very slow)
         nl_source='relu',
         nl_target='relu',
         kqnorm=False,
         save_to=None,
         depth_factor=1.0,             # Scale the depth by this amount
         freeze_blocks=8,
         unfreeze_time = 10_000,       # Number of instances to wait before unfreezing the pro
         loglayers = [1,18,22],
         count_flops = False,
         idmask=True,                  # Whether to apply the id mask trick (replacing some output values by the input) --
         sdepth=None,
         subcontext=64,                # Maximum context to look at when sampling
         project='up-scaling',
         echo_emb=1024,
         echo_conn=16,
         echo_var=0.21,
         echo_max_out=16,
         echo_layers=1,                 # Number of layers in the echo state network
         anti_sol_num=0,                # How many anti-Solomonoff strings to generate per batch
         anti_sol_context=64,           # Size of context to use for generating anti-Solomonoff strings
         anti_sol_seed=(2,33),          # Size of the seed to use for AS strings. This also determines the vocab.
                                        # Chosen uniform-random from the given range
         anti_sol_from=0,               # How long to wait (in instances) before starting to generate AS strings
         anti_sol_buffer=False          # If true, add the AS strings to the buffer. If false, add them to the batch.
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

    if depth_factor != 1.0:
        depth = int(depth * depth_factor)

    if source_microbatch_size is None:
        source_microbatch_size = int(round(target_microbatch_size * source_batch_mult))
    buffer_size = int(round(source_microbatch_size * buffer_size_mult))

    heads = max(width//width_per_head, min_heads)
    assert width % heads == 0

    swidth = width if source_width is None else source_width
    sdepth = get_depth(swidth) if sdepth is None else sdepth
    sheads = max(swidth//width_per_head, min_heads)
    assert width % heads == 0

    # Sweep parms. If these are tuples, we sample a random value in between two given extremes or, for discrete-valued
    # parameters, we sample from a range of options.
    weight_mult1 = rsamp(weight_mult1)
    weight_mult2 = rsamp(weight_mult2)
    weight_multb = rsamp(weight_multb)
    temperature = rsamp(temperature)

    idmask = dsamp(idmask)
    source_mask = dsamp(source_mask)

    wdname = name
    wd = wandb.init(
        name=name,
        project=project,
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
                            num_tokens=NUM_TOKENS, nosqrt=not sqrt_attn_scale, output_mult=out_factor, kqnorm=kqnorm,
                            attn_factor=attn_factor, num_progblocks=max(depth - freeze_blocks, 0))
    # -- Note: the first block of layers consists of regular blocks, the rest are prog blocks.

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

    cooldown_rate = 0.5 ** (1/cooldown)
    # -- The cooldown rate is given in the number of instances to halve the learning rate over. This is the resulting
    #    multiplier per instance.
    last_cooldown = warmup

    print(opt)

    generator = None
    # -- generator(batch_size) generates a batch of source data.

    if source == 'transformer':

        if sequential:
            cmp_source = up.ConditionalTransformer(emb=swidth, heads=sheads, depth=sdepth, seq_length=context,
                        num_tokens=NUM_TOKENS)
        else:
            cmp_source = up.GTransformer(emb=swidth, heads=sheads, depth=sdepth, seq_length=context,
                                     num_tokens=NUM_TOKENS, nl=nl(nl_source), mask_channel=True)

        if torch.cuda.is_available():
            cmp_source.cuda()

        if dp:
            cmp_source = torch.nn.DataParallel(cmp_source)

        buffer = torch.randint(low=0, high=NUM_TOKENS, size=(buffer_size, context), device=d())

        sampletime = -1.0

        def generator_trf(bs):

            # Sample noise from a random model and insert into the buffer
            tic()
            with torch.no_grad():
                # Re-initialize the source
                if old_init:
                    up.weights_init(cmp_source, init_mult_max=50.0, mask_prob_max=0.7)
                else:
                    if sequential:
                        up.weights_init_mup_seq(cmp_source, mult1=weight_mult1, mult2=weight_mult2, multb=weight_multb,
                                            mask=source_mask)
                    else:
                        up.weights_init_mup(cmp_source, mult1=weight_mult1, mult2=weight_mult2, multb=weight_multb,
                                        mask=source_mask)

                # slice a random selection of rows from the buffer (without replacement)
                iz = random.sample(range(buffer.size(0)), source_microbatch_size)
                z = buffer[iz, :]

                # replace some random rows with uniform random characters
                rows = torch.bernoulli(torch.full(size=(source_microbatch_size, 1), fill_value=reset_prob))
                mask = rows.expand(source_microbatch_size, context).to(torch.bool)

                uniform = torch.randint(low=0, high=NUM_TOKENS, size=(source_microbatch_size, context), device=d())
                z[mask] = uniform[mask]

                if sequential:
                    seed = torch.randint(low=0, high=NUM_TOKENS, size=(source_microbatch_size, 1), device=d())
                    batch = sample_sequence(cmp_source, seed, max_context=subcontext, num_tokens=NUM_TOKENS, length=context,
                                            temperature=temperature,
                                            conditional=z, verbose=False)
                    z = batch[:, :-1]
                else:
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
                    if idmask:
                        mask = torch.sigmoid(mask).to(torch.bool)
                        z[mask] = chars[mask]
                    else:
                        z = chars # ignore mask

                buffer[iz, :] = z

                if anti_sol_buffer and anti_sol_num > 0 and instances_seen > anti_sol_from:
                    # Generate some anti-Solomonoff instances.
                    antisol_batch(model, batch=buffer, num=anti_sol_num, context=anti_sol_context,
                                  verbose=random.random() < 0.001, seed_length=random.randrange(*anti_sol_seed),
                                  numchars=NUM_TOKENS, use_mask=True)

                # Now slice a separate sample of instances from the buffer.
                iz = random.sample(range(buffer_size), bs)
                batch = buffer[iz, :]
                # -- Using different indices for the source model and the batch makes the sample more like an iid. sample
                #    (or at least less obviously dependent).

                return batch

        generator = generator_trf

    elif source == 'echo': # Echo-state network

        # cmp_source = up.ReservoirNet(emb=echo_emb, conn=echo_conn, num_tokens=NUM_TOKENS,
        #                              max_out=echo_max_out, init_var=echo_var, nl=torch.tanh)

        # if torch.cuda.is_available(): # The reservoir net is more efficient on CPU (bad implementation, I think)
        #     cmp_source.cuda()

        buffer = torch.randint(low=0, high=NUM_TOKENS, size=(buffer_size, context), device='cpu')

        sampletime = -1.0

        def generator_echo(bs):

            # Sample noise from a random model and insert into the buffer
            tic()
            with torch.no_grad():
                cmp_source = up.ReservoirNet(emb=echo_emb, conn=echo_conn, num_tokens=NUM_TOKENS,
                                             max_out=echo_max_out, init_var=echo_var, nl=torch.tanh, layers=echo_layers)

                # slice a random selection of rows from the buffer (without replacement)
                iz = random.sample(range(buffer.size(0)), source_microbatch_size)
                z = buffer[iz, :]

                # replace some random rows with uniform random characters
                rows = torch.bernoulli(torch.full(size=(source_microbatch_size, 1), fill_value=reset_prob))
                mask = rows.expand(source_microbatch_size, context).to(torch.bool)

                uniform = torch.randint(low=0, high=NUM_TOKENS, size=(source_microbatch_size, context), device='cpu')
                z[mask] = uniform[mask]

                # pass it through a randomly chosen model
                output = cmp_source(z)
                chars = sample(output, temperature=temperature)
                buffer[iz, :] = chars

                # Now slice a separate sample of instances from the buffer.
                iz = random.sample(range(buffer_size), bs)
                batch = buffer[iz, :]
                # -- Using different indices for the source model and the batch makes the sample more like an iid. sample
                #    (or at least less obviously dependent).

                return batch

        generator = generator_echo

    elif source == 'uniform':
        # Simple uniform random noise (for ablations)
        def generator_uniform(bs):
            return torch.randint(low=0, high=NUM_TOKENS, size=(bs, context), device=d())

        generator = generator_uniform

    elif source == 'pointwise':

        def generator_pointwise(bs):

            # The Dirichlet parameters are all near 0, except for a random number which are set to 1.0
            dir_parms = torch.full(fill_value = 1e-8, size=(bs, NUM_TOKENS), dtype=torch.float, device=d())
            for i in range(bs):
                k = random.choice(range(20)) #random.choice(range(NUM_TOKENS))
                mask = random.sample(population=range(NUM_TOKENS), k=k)
                dir_parms[i, mask] = 1.0

            dir = torch.distributions.Dirichlet(dir_parms)

            cat_parms = dir.sample()
            assert cat_parms.size() == (bs, NUM_TOKENS)

            cat = torch.distributions.Categorical(probs=cat_parms)
            sample = cat.sample((context, )).transpose(0, 1)

            return sample

        generator = generator_pointwise

    elif source == 'ndfa':

        def generator_ndfa(bs):
            batch = []
            for i in range(bs):
                batch.append(gen_autseq(aut=None, length=context, vocab=NUM_TOKENS))
            batch = torch.tensor(batch, device=d())

            return batch

        generator = generator_ndfa

    else:
        raise Exception(f'Source {source} not recognized')

    if count_flops:
        # Measure flops per batch
        target_flops = get_flops(model, batch_size=target_microbatch_size, ctx=context, backward=True)
        source_flops = get_flops(cmp_source, batch_size=source_microbatch_size, ctx=context, backward=False)
        print(f'target (GFLOps): {target_flops / 1e9}, source (GFLOps): {source_flops / 1e9}')


    results = { # All relevant results, to be saved as a json file after each eval.
        'vals' : {},
        'locals' : locals()
    }

    for k in datasets.keys():
        results['vals'][k] = {
            'instances' : [],
            'bits' :  [],
            'microbatches' : [],
        }

    accumulated = 0 # nr of instances accumulated currently
    if mbwarmup > 0:
        mbraw = mb_min # macrobatch size as a float
        mbdelta = (macrobatch_size - mb_min) / mbwarmup
    else:
        mbraw = macrobatch_size

    instances_seen = 0
    last_eval = float('-inf')
    last_unfrozen = freeze_blocks - 1

    print('Start pre-training')
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

                results['vals'][name]['instances'].append(instances_seen)
                results['vals'][name]['bits'].append(est)
                results['vals'][name]['microbatches'].append(i)

            last_eval = instances_seen
            with open(f'./{wdname}.json', 'w') as f:
                json.dump(results, f, indent=6, default=lambda o: '<not serializable>') # the json is dumped and overwritten every eval

        sampletime = toc()

        tic()
        # Perform a training step on batches sampled from the buffer

        bs = min(int(round(mbraw)), target_microbatch_size)
        # If the current macrobatch sizer is smaller than the microbatch size, we go with the smaller value
        # (i.e. we leave memory empty).

        batch = generator(bs)

        if not anti_sol_buffer and anti_sol_num > 0 and instances_seen > anti_sol_from:
            # Generate some anti-Solomonoff instances.
            antisol_batch(model, batch=batch, num=anti_sol_num, context=anti_sol_context,
                          verbose=random.random() < 0.001, seed_length=random.randrange(*anti_sol_seed),
                          numchars=NUM_TOKENS, use_mask=True)

        if torch.cuda.is_available():
            batch = batch.cuda()

        input  = batch[:, :-1]
        target = batch[:, 1:]

        with torch.cuda.amp.autocast():
            output = model(input)
            rloss = F.cross_entropy(output.transpose(2, 1), target, reduction='sum')

            loss = (rloss / input.size(1))
            # -- We divide out the time, but sum over the instances

        scaler.scale(loss).backward()
        accumulated += input.size(0)

        if accumulated >= mbraw: # perform a step

            scaler.unscale_(opt)

            # scale the gradients to average over the macrobatch
            # -- here we divide out the instances
            for parm in model.parameters():
                if parm.grad is not None:
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
            'loss': rloss.item() / (input.size(0) * input.size(1)),
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

        if freeze_blocks >  0:
            for il in loglayers:
                if il < depth and type(model.tblocks[il]) is ProgTransformerBlock:
                    wandb.log({
                        f'sig(a) (layer {il})': model.tblocks[il].a.item()
                    }, step=instances_seen)

        wandb.log({}, step=instances_seen, commit=True)

        bar.set_postfix({'loss': f'{rloss.item():.02}'})

        if mbwarmup > 0 and mbraw < macrobatch_size and instances_seen > mb_start:
            mbraw += mbdelta * batch.size(0)

        if warmup > 0 and instances_seen <= warmup:
            for g in opt.param_groups:
                if g['lr'] < g['max_lr']:
                    g['lr'] += g['lr_delta'] * batch.size(0)

        if cooldown > 0 and instances_seen > warmup:
            for g in opt.param_groups:
                g['lr'] *= cooldown_rate ** batch.size(0)

        if i % print_every == 0:

            print('target samples', i)
            for inst in batch[:4, :]:
                inst = remap(inst.tolist())
                print(''.join(str(c) for c in inst))
            print('\n\n')

        if freeze_blocks >  0:
            if last_unfrozen < depth:
                if (instances_seen/unfreeze_time) > ((last_unfrozen+1) / freeze_blocks):
                    print(f'{instances_seen=} unfreezing blocks from {last_unfrozen+1} to {last_unfrozen + freeze_blocks}.')

                    model.enable_layers(lambda j : last_unfrozen + 1 <= j <= last_unfrozen + freeze_blocks)

                    last_unfrozen += freeze_blocks

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