
import up
from up.util import tic, toc, coords, d, sample, sample_sequence, gradient_norm
from up.data import load_data, cas, repeval

from former.util import d, here, tic, toc, sample_batch, enwik8_string, enwik8_bytes, estimate_compression
import former

import wandb, random, fire, gzip

import torch
from torch import nn
import torch.nn.functional as F

import math, json
from math import floor, sqrt

from tqdm import trange

"""
Load a UP pretrained model and train it on some data. 

"""

NUM_TOKENS = 256
LOG2E = math.log2(math.e)
LOGE2 = math.log(2.0)
REPS = [1, 3, 10] # rep evaluation
LOGLAYERS = [1, 18, 22]

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

def go(checkpoint,
         target_data='wp-train', # Which target dataset to finetune on
         lr=None,
         lr_mult=1.0, # Multiplier for the learning rate, only applies to non-baseline model, only if warmup is turned on
         num_batches=600_000,
         wdname='data',
         wandb_project='up-data',
         eval_ood=True,   # evaluate on the OOD data
         eval_test=False, # Evaluate on test in addition to val
         debug=False,
         warmup=100_000,
         eval_every=5_000,
         print_every=500_000,
         gc=1.0,
         mbwarmup = 100_000,   # Accumulation warmup (in instances)
         mb_min = 16,          # Minimum microbatch size to start warming up from.
         mb_start = 100_000,   # Number of instances to wait before starting the warmup
         cooldown=10_000_000,  # After the warmup finishes, we cool down by halving the lr every `cooldown` instances
         sequential=False, eval_samples=10_000,
         baseline=False, # If true, use the random initialization of the model rather than the saved parms
         skip_eval=False,  # Whether to skip the evaluation
         eval_batch_mult=2.0,  # How much bigger the eval batches can be than the training batches
         train_prop=1.0,  # Proportion of the training data to use
         context=None,    # override for the context
         bs=None,         # override for the batch size
         reset_opt=False, # reset the optimizer, rather than reusing it.
       ):
    """
    Load a UP checkpoint and train
    """
    if context is not None:
        assert baseline or reset_opt
        # We can only extend the context window by replacing the embedding layer, which requires resetting the optimizer

    cp = torch.load(checkpoint, map_location=None if torch.cuda.is_available() else torch.device('cpu'))

    hp = cp['locals'] # hyperparams

    microbatch_size = bs if bs is not None else hp['target_microbatch_size']
    ctx = context if context is not None else hp['context']
    macrobatch_size = hp['macrobatch_size']

    localvars = locals()
    wd = wandb.init(
        name=wdname,
        project=wandb_project,
        config=localvars,
        mode= 'disabled' if debug else 'online'
    )

    # Target for training
    model = up.GTransformer(emb=hp['width'], heads=hp['heads'], depth=hp['depth'], seq_length=hp['context'], nl=nl(hp['nl_target']),
                            num_tokens=NUM_TOKENS, nosqrt=not hp['sqrt_attn_scale'], output_mult=hp['out_factor'], kqnorm=hp['kqnorm'],
                            attn_factor=hp['attn_factor'], num_progblocks=max(hp['depth'] - hp['freeze_blocks'], 0))
    # -- Note: the first block of layers consists of regular blocks, the rest are frozen blocks (these are progressively unfrozen)
    #    as training progresses.
    # -- Note: we have to load the original cpontext size first and then extend, or the state-dict won't match

    if torch.cuda.is_available():
        model.cuda()

    opt = model.mup(base_lr=lr, width0=hp['width0'], factor=hp['init_factor'], optcls=torch.optim.AdamW, weight_decay=hp['weight_decay'])
    print('MUP-initialized model.')

    if baseline or reset_opt:
        if warmup > 0:
            for g in opt.param_groups:
                g['max_lr'] = g['lr']
                g['lr_delta'] = g['lr'] / warmup

                g['lr'] = 0.0

    if not baseline:
        model.load_state_dict(cp['model_state_dict'])
        if not reset_opt:
            opt.load_state_dict(cp['optimizer_state_dict'])
            # NB: State dict includes the learning rate and weight decay, so they are taken from the checkpoint, NOT the command line parms.

        print('optimizer setup (just after loading)')
        for g in opt.param_groups:
            print(f"   {g['max_lr']=} {g['lr']=} {g['lr_delta']=} {warmup=} ")

        if warmup > 0:
            for g in opt.param_groups:
                g['lr_delta'] = (g['max_lr'] * lr_mult) / warmup
                g['lr'] = 0.0

        print('Pretraining run, loaded model/opt state dict.')

    if context is not None:
        # Replace the position embeddings
        model.pos_embedding = nn.Embedding(embedding_dim=hp['width'], num_embeddings=context)
        if torch.cuda.is_available():
            model.pos_embedding.cuda()

        # Create a new optimizer (using the MUP logic)
        opt = model.mup_opt(base_lr=lr, width0=hp['width0'], factor=hp['init_factor'], optcls=torch.optim.AdamW, weight_decay=hp['weight_decay'])

        # Set the warmup and lr_mult
        if warmup > 0:
            for g in opt.param_groups:
                g['max_lr'] = g['lr']
                g['lr_delta'] = (g['max_lr'] * lr_mult) / warmup
                g['lr'] = 0.0

    print('optimizer setup')
    for g in opt.param_groups:
        print(f"   {g['max_lr']=} {g['lr']=} {g['lr_delta']=} {warmup=} ")

    if eval_ood:
        datasets = {
            'champ': torch.tensor(load_data('champ'), dtype=torch.long),
            'dyck' : torch.tensor(load_data('dyck'), dtype=torch.long),
            'ndfa' : torch.tensor(load_data('ndfa'), dtype=torch.long),
            'toy'  : torch.tensor(load_data('toy'),  dtype=torch.long),
            'bitsrep' : torch.tensor(load_data('bits'), dtype=torch.long),
            'wp'   : torch.tensor(load_data('wp-val'), dtype=torch.long),
            'linux': torch.tensor(load_data('linux-val'), dtype=torch.long)
        }
    else:
        datasets = {
            'wp'   : torch.tensor(load_data('wp-val'), dtype=torch.long)
        }

    if eval_test:
        testsets = {
            'wp'       : torch.tensor(load_data('wp-test'), dtype=torch.long),
            'german'   : torch.tensor(load_data('german-test'), dtype=torch.long),
            'aut'      : torch.tensor(load_data('aut'), dtype=torch.long),
            'toy2'     : torch.tensor(load_data('toy2'), dtype=torch.long),
            'bitsflip' : torch.tensor(load_data('bitsflip'),  dtype=torch.long),
            'code'     : torch.tensor(load_data('code-test'), dtype=torch.long),
        }
    else:
        testsets = {}

    scaler = torch.cuda.amp.GradScaler()

    results = { # All relevant results, to be saved as a json file after each eval.
        'vals' : {},
        'tests' : {},
        'locals' : localvars
    }

    for name in list(datasets.keys()) + [f'rep-{r}' for r in REPS]:
        results['vals'][name] = {
            'instances' : [],
            'bits' :  [],
            'microbatches' : [],
        }

    for name in list(testsets.keys()):
        results['tests'][name] = {
            'instances' : [],
            'bits' :  [],
            'microbatches' : [],
        }

    # Fine-tuning
    print('Start data-training')
    traindata = torch.tensor(up.data.load_data(target_data), device='cpu')

    if train_prop < 1.0:
        # Train only on a small percentage of the data

        newsize = int(round(traindata.size(0) * train_prop))
        traindata = traindata[:newsize]

    cooldown_rate = 0.5 ** (1/cooldown)
    # -- The cooldown rate is given in the number of instances to halve the learning rate over. This is the resulting
    #    multiplier per instance.
    last_cooldown = warmup

    accumulated = 0 # nr of instances accumulated currently
    if mbwarmup > 0:
        mbraw = mb_min # macrobatch size as a float
        mbdelta = (macrobatch_size - mb_min) / mbwarmup
    else:
        mbraw = macrobatch_size

    instances_seen = 0
    last_eval = float('-inf')
    last_unfrozen = hp['freeze_blocks'] - 1

    for i in (bar := trange(num_batches)):

        if instances_seen - last_eval > eval_every and not skip_eval:
            valbs = int(microbatch_size * eval_batch_mult)
            # Evaluate on simple repeated patterns
            for r in REPS:
                print(f'evaluating rep {r}')

                with torch.no_grad():
                    bits = repeval(model=model, context=ctx, rep=r, num_tokens=NUM_TOKENS,
                                   batch_size=valbs, nbatches=round(eval_samples/valbs))

                name = f'rep-{r}'
                wandb.log({f'rval/{name}': bits}, step=instances_seen)

                results['vals'][name]['instances'].append(instances_seen)
                results['vals'][name]['bits'].append(bits)
                results['vals'][name]['microbatches'].append(i)

            for name, data in datasets.items():
                print(f'evaluating {name}')

                with torch.no_grad():
                    est = estimate_compression(
                        model=model,
                        data=data,
                        nsamples=eval_samples,
                        context=ctx,
                        batch_size=valbs,
                        model_produces_logits=True
                    )

                wandb.log({f'val/{name}': est}, step=instances_seen)

                results['vals'][name]['instances'].append(instances_seen)
                results['vals'][name]['bits'].append(est)
                results['vals'][name]['microbatches'].append(i)

            if eval_test:
                for name, data in testsets.items():
                    print(f'evaluating {name}')

                    with torch.no_grad():
                        est = estimate_compression(
                            model=model,
                            data=data,
                            nsamples=eval_samples,
                            context=ctx,
                            batch_size=valbs,
                            model_produces_logits=True
                        )

                    wandb.log({f'test/{name}': est}, step=instances_seen)

                    results['tests'][name]['instances'].append(instances_seen)
                    results['tests'][name]['bits'].append(est)
                    results['tests'][name]['microbatches'].append(i)

            last_eval = instances_seen
            with open(f'./{wdname}.json', 'w') as f:
                json.dump(results, f, indent=6, default=lambda o: '<not serializable>') # the json is dumped and overwritten every eval

        # sample a batch from the data
        source, target = sample_batch(traindata, ctx, microbatch_size)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        with torch.cuda.amp.autocast():
            output = model(source)
            rloss = F.cross_entropy(output.transpose(2, 1), target, reduction='sum')

            loss = (rloss / source.size(1))

        scaler.scale(loss).backward()
        accumulated += source.size(0)

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

        ### Admin
        wandb.log({
            'loss': rloss.item() / (source.size(0) * source.size(1)),
            'pre-training': 1.0,
            'mbraw': mbraw
        }, step=instances_seen)

        wandb.log({
            'learning_rate (stable)': opt.param_groups[0]['lr'],
            'learning_rate (scaled)': opt.param_groups[1]['lr'],
        }, step=instances_seen)

        if hp['freeze_blocks'] >  0:
            for il in LOGLAYERS:
                if il < hp['depth'] and type(model.tblocks[il]) is up.ProgTransformerBlock:
                    wandb.log({
                        f'sig(a) (layer {il})': model.tblocks[il].a.item()
                    }, step=instances_seen)

        wandb.log({}, step=instances_seen, commit=True)

        bar.set_postfix({'loss': f'{rloss.item():.02}'})

        if mbwarmup > 0 and mbraw < macrobatch_size and instances_seen > mb_start:
            mbraw += mbdelta * source.size(0)

        if warmup > 0 and instances_seen <= warmup:
            for g in opt.param_groups:
                if g['lr'] < g['max_lr']:
                    g['lr'] += g['lr_delta'] * source.size(0)

        if cooldown > 0 and instances_seen > warmup:
            for g in opt.param_groups:
                g['lr'] *= cooldown_rate ** source.size(0)

        if hp['freeze_blocks'] >  0:
            if last_unfrozen < hp['depth']:
                if (instances_seen/hp['unfreeze_time']) > ((last_unfrozen+1) / hp['freeze_blocks']):
                    print(f'{instances_seen=} unfreezing blocks from {last_unfrozen+1} to {last_unfrozen + hp["freeze_blocks"]}.')

                    model.enable_layers(lambda j : last_unfrozen + 1 <= j <= last_unfrozen + hp['freeze_blocks'])

                    last_unfrozen += hp['freeze_blocks']

        instances_seen += source.size(0)


def set_lr(lr, opt):
    for g in opt.param_groups:
        g['lr'] = lr
        g['initial_lr'] = lr

def grad_scale(model, scale):
    for parm in model.parameters():
        parm.grad *= scale

if __name__ == '__main__':
    fire.Fire()