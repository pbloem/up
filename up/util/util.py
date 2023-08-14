import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

from collections import Counter
from collections.abc import Iterable

import os, math, tqdm, time
from tqdm import trange

import warnings

tics = []

def tic():
    tics.append(time.time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time.time()-tics.pop()

def log2(x):
    return math.log(x) / math.log(2.0)

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def kl_loss(zmean, zsig):
    b, l = zmean.size()

    kl = 0.5 * torch.sum(zsig.exp() - zsig + zmean.pow(2) - 1, dim=1)
    # -- The KL divergence between a given normal distribution and a standard normal distribution
    #    can be rewritten this way. It's a good exercise to work this out.

    assert kl.size() == (b,)
    # -- At this point we want the loss to be a single value of each instance in the batch.
    #    Asserts like this are a good way to document what you know about the shape of the
    #    tensors you deal with.

    return kl

def vae_sample(zmean, zsig):
    b, l = zmean.size()

    # sample epsilon from a standard normal distribution
    eps = torch.randn(b, l, device=d())

    # transform eps to a sample from the given distribution
    return zmean + eps * (zsig * 0.5).exp()

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view( (input.size(0),) + self.shape) # keep the batch dimensions, reshape the rest

class Lambda(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, input):
        return self.func(input)


def coords(h, w):

    xs = torch.arange(h, device=d())[None, None, :, None] / h
    ys = torch.arange(w, device=d())[None, None, None, :] / w
    xs, ys = xs.expand(1, 1, h, w), ys.expand(1, 1, h, w)
    res = torch.cat((xs, ys), dim=1)

    assert res.size() == (1, 2, h, w)

    return res

class TrieNode():

    def __init__(self):
        self.count = 0
        self.children = {}

    def add(self, tokens):
        self.count += 1

        if len(tokens) > 0:
            c = tokens[0]

            if c not in self.children:
                self.children[c] = TrieNode()

            child = self.children[c]
            child.count += 1
            child.add(tokens[1:])

    def get_count(self, tokens):
        if len(tokens) == 0:
            return self.count

        c = tokens[0]
        if c not in self.children:
            return 0

        child = self.children[c]
        return child.get_count(tokens[1:])

class Trie():

    def __init__(self):
        self.root = TrieNode()

    def add(self, tokens:str):
        self.root.add(tokens)

    def get_count(self, tokens):
        return self.root.get_count(tokens)

    # -- Should be even faster to get all n-gram counts in one sweep. Requires storing reversed n-grams


def markov(train : str, val:str, test :str, max_order=3, lambdas=[1.0, 0.1 , 0.01, 0.0001, 1e-6], numtokens=None, verbose=False):
    """
    Computes the compression length of the test data under all Markov models up to the given order.

    :param train:
    :param test:
    :param order:
    :param laplace: The lambda parameter for the laplace smoothing.
    :return:
    """
    ran = tqdm.trange if verbose else range

    models = [Counter() for o in range(max_order + 1)]
    # -- The 0 order model counts 1-grams, the 1 order model counts 2-grams and so on. The order is the
    #    index.

    if verbose:
        print('Creating frequency models.')
        tic()

    for i in ran(len(train)):
        for order, model in enumerate(models):
            if i >= order:
                ngram = train[i-(order):i+1]
                assert len(ngram) == order + 1, f'{i=}, {order=}'

                model[ngram] += 1

        # trie.add(train[i - max_order:i+1])

    if verbose: print(f'done ({toc():.4}s).')

    numtokens = len(models[0]) if numtokens is None else numtokens

    if verbose: print('Choosing smoothing levels.')
    res_val = []

    for i in ran(len(lambdas)):
        l = lambdas[i]
        res = codelength(models, val, len(train), numtokens=numtokens, smoothing=l, verbose=False)
        res_val.append(res)

    matrix = torch.tensor(res_val)
    # -- this has the lambdas vertically and the order horizontally
    lambda_indices = matrix.argmin(dim=0)
    smoothing = [lambdas[i] for i in lambda_indices]
    if verbose: print('smoothing levels chosen: ', smoothing)

    # Compute test set codelengths
    if verbose:
        print('Computing codelengths.')
        tic()

    res = codelength(models, test, len(train), numtokens=numtokens, smoothing=smoothing, verbose=verbose)

    if verbose: print(f'done ({toc():.4}s).')

    res = [r / len(test) for r in res]
    return res

def codelength(models, data, len_train, numtokens, smoothing, verbose=False):

    ran = tqdm.trange if verbose else range

    res = [0.0] * len(models)

    if type(smoothing) is float:
        smoothing = [smoothing] * len(models)

    for i in ran(len(data)):
        lprob = None
        for order, model in enumerate(models):
            if i >= order:
                ngram = data[i-(order):i+1]
                cond  = ngram[:-1]
                # -- The tokens preceding the token for which we're estimating the probability. That is, the conditional
                #    in p(x_i | x_0, ..., x_{i-1}).

                denom = len_train if cond == '' else models[order-1][cond]
                lprob = log2(models[order][ngram] + smoothing[order]) - log2(denom + smoothing[order] * numtokens)

            res[order] += - lprob
            # -- Note the sneaky trick here. If the window is large enough for the current order of model, we compute
            #    the regular probability. However, if we are early in the sequence, and we can't have a full n-gram yet,
            #    we just revert to whatever the previous model computed as an estimate.

    return res



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

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

def find_batch_size(model, loss, input, opt=None, upper=2000, samples=10, burn_in=10, wandb=None):
    """
    Runs a fibonacci search over batch sizes to find the one with the highest throughput

    :param model:
    :param input: Either a single batch or an iterable of batches of the shape and datatype the model expects. The batch
    dimension should be 1.
    :param shape:
    :param loss: A function that takes the model output and computes a loss. This may contain a dummy target variable. If
    the model and batch are cuda then this dummy variable should be as well.
    :return:
    """

    if isinstance(input, torch.Tensor):
        input = (input,)

    assert all(i.size(0) == 1 for i in input), str(list(i.size() for i in input))

    if opt is None:
        opt = torch.optim.Adam(params=model.parameters(), lr=3e-4)

    search = Search(
        lambda b : throughput(b, model, loss, input, opt, samples=samples, burn_in=burn_in), max_x=upper)

    if all(y == float('inf') for y in search.y):
        raise Exception(f'All batch sizes led to out-of-memory errors. Batch sizes sampled: {search.x}. Throughputs: {search.y}')

    if search.opt == upper:
        warnings.warn('The best batch size found was the upper bound of the search interval. You may want to try again with a higher value for `upper`.')

    if wandb is not None:
        table = wandb.Table(data=[[x, y] for (x, y) in zip(search.x, search.y)],
                            columns=['batch_sizes', 'throughputs'])
        plot = wandb.plot.line(table=table, x='batch_sizes', y='throughputs', title='throughput test')
        wandb.log({'throughput test': plot})

    return search.opt, search.x, search.y

class Search():
    """
    A fibonacci search.
    """

    def __init__(self, function, max_x):
        """

        :param function: A function `f(x : int)` to maximize.
        """
        self.function = function
        self.samples = {} # inputs probed

        self.max_x = max_x if FIB.is_fibonacci(max_x) else FIB.next(max_x)

        self.search(0, self.max_x, 0)

        self.x = [k for k, _ in self.samples.items()]
        self.y = [v for _, v in self.samples.items()]

        self.opt, opty = -1, float('-inf')
        for x, y in self.samples.items():
            if y > opty:
                opty = y
                self.opt = x

    def probe(self, x):
        if x not in self.samples:
            y = self.function(x)
            self.samples[x] = y
        else:
            y = self.samples[x]

        return y

    def search(self, fr:int, to:int, depth:int):
        print(f'-- testing interval ({fr}, {to}) (recursion depth {depth})')

        range = to - fr

        if range <= 2:
            # -- base case: `fr` and `to` are neighbouring integers
            self.probe(fr)
            self.probe(fr+1)
            self.probe(to)
            return

        # -- cut the interval with two midpoints
        prev = FIB.previous(range)
        mid0 = to - prev
        mid1 = fr + prev

        assert (mid0 in self.samples) or (mid1 in self.samples), f'{self.samples=}'
        # -- The key idea of the Fibonacci search is that one of these points we will have seen before.

        y0 = self.probe(mid0)
        y1 = self.probe(mid1)

        if y0 > y1:
            self.search(fr, mid1, depth + 1)
        else:
            self.search(mid0, to, depth + 1)


def throughput(batch_size, model, loss, input, opt, samples=10, burn_in=10):
    """
    Returns the throughput in number of instances per second.

    :param model:
    :param input:
    :param samples:
    :return:
    """

    if batch_size < 1:
        return float('-inf')

    try :
        batch = [i.expand(batch_size, *i.size()[1:]).contiguous() for i in input]

        for _ in range(burn_in):
            opt.zero_grad()

            output = model(*batch)
            l = loss(output)
            l.backward()
            opt.step()

        tic()
        for _ in range(samples):
            opt.zero_grad()

            output = model(*batch)
            l = loss(output)
            l.backward()
            opt.step()

        total_time = toc()
        total_instances = samples * batch_size

        return total_instances / total_time

    except torch.cuda.OutOfMemoryError as e:
        # message = getattr(e, 'message', repr(e))
        # if 'memory' not in message:
        #     print(f'Runtime error for batch size {batch_size}. Treating this as out-of-memory (i.e. infinite '
        #                   f'throughput), but it seems to be something else. Please inspect the error below.')
        #     print(e)
        # else:
        #     print(f'OOM (error caught {message}).')
        torch.cuda.empty_cache()

        return float('-inf')

SQRT5 = math.sqrt(5)
PHI = 1.61803398874989484820

class Fibonacci():
    """
    Utility class for retrieving Fibonacci numbers easily.
    """

    def __init__(self, max_index=92):

        self.numbers = [0, 1]
        for i in range(2, max_index + 1):
            self.numbers.append(self.numbers[-1] + self.numbers[-2])

    def is_fibonacci(self, n:int):
        """
        :param n:
        :return: True if `n` is a Fibonacci number

        """
        s = 5 * n * n

        return is_square(s + 4) or is_square(s-4)

    def get_index(self, n:int):
        """
        :param n:
        :return: Return the index of a given Fibonacci `n`. If `n` is not a Fibonacci number, the index of the nearest
         Fibonacci number is returned
        """

        if n == 0: return 0

        if n == 1: return 2

        return int(round(self.get_index_approx(n)))

    def get_index_approx(self, n :int):
        """
        Returns the approximate index of the given number. If the number is not a
	    fibonacci number, a non-integer value is returned indicating the two
	    nearest fibonacci numbers (ie. if the returned value is 33.2, the number
	    is above the 33rd fibonacci number and below the 34th).

        :param n:
        :return:
        """
        return math.log(n * SQRT5 + 0.5)/math.log(PHI)

    def previous(self, n : int):
        """
        :param n:
        :return: The previous Fibonacci number before `n`.
        """
        i = self.get_index(n)
        return 0 if i == 0 else self.numbers[i-1]

    def next(self, n : int):
        """
        :param n:
        :return: The next Fibonacci number after `n`.
        """

        i = self.get_index(n)
        return self.numbers[i+1]

FIB = Fibonacci()

def is_square(n : int):
    """
    Check if a given integer is square. This could be optimized if necessary:
    https://stackoverflow.com/questions/295579/fastest-way-to-determine-if-an-integers-square-root-is-an-integer
    :param n:
    :return:
    """
    if n < 0:
        return False

    if n == 0:
        return True

    x, y = 1, n
    while x + 1 < y:
        mid = (x+y)//2
        if mid**2 < n:
            x = mid
        else:
            y = mid

    return n == x**2 or n == (x+1)**2


