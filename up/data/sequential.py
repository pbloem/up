import string

import wget, os, gzip, pickle, random, re, sys

from former.util import enwik8_bytes

IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.{}.pkl.gz'

PAD, START, END, UNK = '.pad', '.start', '.end', '.unk'

SENT = '_s'

TOY = {
    '_s': ['_s _adv', '_np _vp', '_np _vp _prep _np', '_np _vp ( _prep _np )', '_np _vp _con _s',
           '_np _vp ( _con _s )'],
    '_adv': ['briefly', 'quickly', 'impatiently'],
    '_np': ['a _noun', 'the _noun', 'a _adj _noun', 'the _adj _noun'],
    '_prep': ['on', 'with', 'to'],
    '_con': ['while', 'but'],
    '_noun': ['mouse', 'bunny', 'cat', 'dog', 'man', 'woman', 'person'],
    '_vp': ['walked', 'walks', 'ran', 'runs', 'goes', 'went'],
    '_adj': ['short', 'quick', 'busy', 'nice', 'gorgeous']
}

"""
The test set performance (in bits-per-byte) of a n-th order Markov model with 0.01 Laplace smoothing. See experiments/markov
for the code that was used to compute these.

MARKOV[name][n] represents the n-th order markov model for dataset `name`. For the generated datasets, separate datasets 
were generated for train, test and val, and for wikipedia, the canonical splits were used.   
"""
MARKOV = {
    'wp' : [5.106629000536541, 3.920734110007795, 3.0992338340343406, 2.5360331955140865, 2.2364564738213955, 2.1894437828412445],
    'dyck' : [1.37402480190898, 1.1336195970119751, 1.133641671511376, 0.9372411063969551, 0.9372486322251692, 0.8904998264766827],
    'ndfa' : [3.4569746915375603, 0.6658862666099562, 0.2942196511608079, 0.2942189625751208, 0.2942892796818609, 0.29438668804011764],
    'toy' : [4.196220178391961, 2.2355184935857064, 0.9741857673348161, 0.7252455261062091, 0.6389911594450575, 0.6320596343654071],
    'bits': [1.1733818782355905, 1.1691583755645572, 1.166750905432737, 1.164872046306139, 1.1631674006093808, 1.1381807911247805],
    'champ': [4.020301090644664, 4.182040925227881, 4.475985937327467, 4.309589578934016, 4.104379828690582, 4.090988560452365],
    'bitsrep': [1.000061690176759, 1.0001037284531082, 1.0001749039272738, 0.6483094648040643, 0.6241204968147399, 0.6224601673202647],
}

"""
Context-based Markov model performance.
"""
MARKOV_CTX = {
    'wp' : [4.9800, 4.8119, 5.6298, 6.3050, 6.7800, 7.0469],     # 4.81
    'dyck' : [1.3888, 1.1604, 1.1765, 1.0077, 1.0747, 1.1769],   # 1.0
    'ndfa' : [3.5141, 0.7293, 0.3716, 0.3918, 0.4352, 0.5322],   # 0.37
    'toy' : [4.2825, 2.6090, 1.9711, 2.4323, 2.8710, 3.3567],    # 1.97
    'bits': [1.1875, 1.1961, 1.2226, 1.2693, 1.3434, 1.4328],    # 1.22
    'champ': [3.3676, 2.4247, 2.6952, 3.3094, 3.9252, 4.3293],   # 2.42
    'bitsrep': [1.0057, 1.0127, 1.0215, 0.6922, 0.6957, 0.7379], # 0.62
}

def load_imdb(final=False, val=5000, seed=0, voc=None, char=False):

    cst = 'char' if char else 'word'

    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url)

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}

        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}

        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = []
            for seq in seqs:
                seq = [s if s < mx else unk for s in seq]
                nw_sequences[key].append(seq)

        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    # Make a validation split
    random.seed(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []

    val_ind = set( random.sample(range(len(sequences['train'])), k=val) )
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), \
           (x_val, y_val), \
           (i2w, w2i), 2


def gen_sentence(sent=SENT, g=TOY):

    symb = '_[a-z]*'

    while True:

        match = re.search(symb, sent)
        if match is None:
            return sent

        s = match.span()
        sent = sent[:s[0]] + random.choice(g[sent[s[0]:s[1]]]) + sent[s[1]:]

def gen_dyck(p=7/16):
    open = 1
    sent = '('
    while open > 0:
        if random.random() < p:
            sent += '('
            open += 1
        else:
            sent += ')'
            open -= 1

    return sent

def gen_ndfa(p=1/4):

    word = random.choice(['abc!', 'uvw!', 'klm!'])

    s = ''
    while True:
        if random.random() < p:
            return 's' + s + 's'
        else:
            s+= word

def load_brackets(n=50_000, seed=0):
    return load_toy(n, char=True, seed=seed, name='dyck')

def load_ndfa(n=50_000, seed=0):
    return load_toy(n, char=True, seed=seed, name='ndfa')

def load_toy(n=50_000, char=True, seed=0, name='lang'):

    random.seed(0)

    if name == 'lang':

        sentences = [ gen_sentence(SENT, TOY) for _ in range(n)]
        sentences.sort(key=lambda s : len(s))

    elif name == 'dyck':

        sentences = [gen_dyck(7./16.) for _ in range(n)]
        sentences.sort(key=lambda s: len(s))

    elif name == 'ndfa':

        sentences = [gen_ndfa(1./4.) for _ in range(n)]
        sentences.sort(key=lambda s: len(s))

    else:
        raise Exception(name)

    tokens = set()
    for s in sentences:

        if char:
            for c in s:
                tokens.add(c)
        else:
            for w in s.split():
                tokens.add(w)

    i2t = [PAD, START, END, UNK] + list(tokens)
    t2i = {t:i for i, t in enumerate(i2t)}

    sequences = []
    for s in sentences:
        if char:
            tok = list(s)
        else:
            tok = s.split()
        sequences.append([t2i[t] for t in tok])

    return sequences, (i2t, t2i)

def gen_bits(wordlength=5):
    w1 = [random.choice([True, False]) for _ in range(wordlength)]
    w2 = [random.choice([True, False]) for _ in range(wordlength)]

    wxor = [b1 != b2 for b1, b2 in zip(w1, w2)]
    wand = [b1 and b2 for b1, b2 in zip(w1, w2)]
    wor  = [b1 or b2 for b1, b2 in zip(w1, w2)]
    weq  = [b1 == b2 for b1, b2 in zip(w1, w2)]

    return ''.join('0' if b else '1' for b in (w1 + w2 + wxor + wand + wor + weq))

def gen_bitsrep(wordlength=3, rep=3):
    """
    Returns a random sequence of `wordlength` bits, repeated `rep` times
    :param wordlength:
    :param rep:
    :return:
    """
    word = [random.choice(('0', '1')) for _ in range(wordlength)]
    return ''.join(b for b in (word * rep))

def gen_champ(length=256, mx=16777216):
    """
    Generator inspired by the Champernowne constant 0.123456789101112...

    We simply concatenate all integers from 0 to n into a sequence.

    The result is a sequence in which each digit and n-gram appears with equal relative frequency (in the limit)
    but which is also entirely predictable. This means that n-gram models should not do better than uniform
    chance, but a model with more computional power should easily compress the whole sequence far below that.

    :param mx:
    :param length:
    :return:
    """
    start = random.choice(range(0, mx-length))

    seq = ''
    for i in range(start, start+length):
        seq += f'{i:x}'

    return seq

def to_bytes(s:str):
    """
    Converts a string to a series of integers between 0 and 256.

    :param s:
    :return:
    """

    res = [ord(ch) for ch in s]
    assert all(ch >= 0 and ch < 256 for ch in res)

    return res

def to_str(ls, printable=True):
    """
    Converts a list of integers representing bytes to a (printable) string.
    :param ls:
    :return:
    """
    res = ''

    for i in ls:
        res += cas(i) if printable else chr(i)

    return res

PRINTABLE = set(string.digits + string.ascii_letters + string.punctuation)
#-- NB we don't print whitespace

def cas(i):
    """
    Character-as-string. Filters out the ascii codes that aren't safe to print.

    :return:
    """
    assert i >= 0 and i < 256
    return '□' if i not in PRINTABLE else str(chr(i))

def load_str(name='dyck', num_chars=100_000, final=False, printable=True):
    """

    :param name:
    :param num_chars:
    :param final:
    :param printable: Ensure that the string is printable (this may collapse some characters into a single one)
    :return:
    """
    res = load_data(name, num_chars, final)
    return to_str(res, printable=printable)

def load_data(name='dyck', num_chars=100_000, final=False, char_offset=0):
    """
    Load a dataset as a single contiguous string.

    :param name:
    :param num_chars: Approximate nr of characters.
    :return:
    """

    # Generators with delimiter
    gen_delim = {
        'dyck' : gen_dyck,
        'ndfa' : gen_ndfa,
        'toy'  : gen_sentence,
        'bits' : gen_bits,
        'champ': gen_champ,
    }

    # Generators without delimiter
    gen_nodelim = {
        'bitsrep' : gen_bitsrep,
    }

    if name in gen_delim.keys():
        res = '|'
        while len(res) < num_chars:
            res += gen_delim[name]()
            res += '|'

        return [c + char_offset for c in to_bytes(res)]

    if name in gen_nodelim.keys():
        res = ''
        while len(res) < num_chars:
            res += gen_nodelim[name]()

        res = [c + char_offset for c in to_bytes(res)]
        return res

    if name == 'wp':
        if final:
            return [int(c) + char_offset for c in enwik8_bytes()[2]] # test data
        else:
            return [int(c) + char_offset for c in enwik8_bytes()[1]] # validation data

    if name == 'wp-train':
        return [int(c) + char_offset for c in enwik8_bytes()[0]]

    if name == 'wp-val':
        return [int(c) + char_offset for c in enwik8_bytes()[1]]

    if name == 'wp-test':
        return [int(c) + char_offset for c in enwik8_bytes()[2]]


def gen_n(p=0.9):

    n = 1
    while random.random() < p:
        n += 1

    #-- This is a bit inefficient
    return n

def gen_aut(num_states=5, num_symbols=3, p=0.3, vocab=256):
    """
    Generate a random non-deterministic automaton.
    :param states:
    :param symbols:
    :param p:
    :return:
    """
    assert num_symbols <= vocab

    symbols = random.sample(k=num_symbols, population=range(vocab))
    states = list(range(num_states))

    edges = []
    for i in states:
        for j in states:
            if random.random() < p:
                edges.append((i, j, random.choice(symbols))) # an edge

    aut = {i: [] for i in states}
    for i, j, s in edges:
        aut[i].append((j, s))

    return aut, symbols

def gen_autseq(aut=None, length=512, vocab=256):

    if aut is None:
        num_states = gen_n()
        num_symbols = min(gen_n(), vocab)
        aut, symbols = gen_aut(num_states=num_states, num_symbols=num_symbols, p=random.random(), vocab=vocab)

    sequence = []
    state = 0

    while len(sequence) < length:
        out_edges = aut[state]
        if len(out_edges) == 0:
            state = 0
            sequence.append(symbols[0])
            # -- If we hit a terminal state, we add the first symbol and reset the state to 0. The addition of a symbol
            #    ensures that every automaton generates a sequence.
        else:
            state, symbol = random.choice(out_edges)

            sequence.append(symbol)

    assert len(sequence) == length

    return sequence

if __name__ == '__main__':
    print(''.join(str(i) for i in gen_autseq(vocab=10)))