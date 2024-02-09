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

MARKOV[name][n] represents the n-th order markov model for dataset `name`. For the generated datasets separate datasets 
were generated for train, test and val, and for wikipedia, the canonical splits were used.   
"""
MARKOV = {
    'wp' : [5.106629000536541, 3.920734110007795, 3.0992338340343406, 2.5360331955140865, 2.2364564738213955, 2.1894437828412445],
    'dyck' : [1.37402480190898, 1.1336195970119751, 1.133641671511376, 0.9372411063969551, 0.9372486322251692, 0.8904998264766827],
    'ndfa' : [3.4569746915375603, 0.6658862666099562, 0.2942196511608079, 0.2942189625751208, 0.2942892796818609, 0.29438668804011764],
    'toy' : [4.196220178391961, 2.2355184935857064, 0.9741857673348161, 0.7252455261062091, 0.6389911594450575, 0.6320596343654071]
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

    return ['0' if b else '1' for b in w1+w2+wxor+wand+wor+weq]

def to_bytes(s:str):
    """
    Converts a string to a series of integers between 0 and 256.

    :param s:
    :return:
    """

    res = [ord(ch) for ch in s]
    assert all(ch >= 0 and ch < 256 for ch in res)

    return res

def to_str(ls):
    """
    Converts a list of integers representing bytes to a (printable) string.
    :param ls:
    :return:
    """
    res = ''

    for i in ls:
        res += cas(i)

    return res

def cas(i):
    """
    Character-as-string. Filters out the ascii codes that aren't safe to print.

    128-160 are automatically printed as a "null" box .
    :return:
    """
    assert i >= 0 and i < 256
    return str(chr(128)) if (i < 33) else str(chr(i))

def load_str(name='dyck', num_chars=100_000, final=False):
    res = load_data(name, num_chars, final)
    return to_str(res)

def load_data(name='dyck', num_chars=100_000, final=False):
    """
    Load a dataset as a single contiguous string.

    :param name:
    :param num_chars: Approximate nr of characters.
    :return:
    """

    gen = {
        'dyck' : gen_dyck,
        'ndfa' : gen_ndfa,
        'toy'  : gen_sentence,
        'bits' : gen_bits
    }

    if name in gen.keys():
        res = '|'
        while len(res) < num_chars:
            res += gen[name]()
            res += '|'

        return to_bytes(res)

    if name == 'wp':
        if final:
            return [int(c) for c in enwik8_bytes()[2]] # test data
        else:
            return [int(c) for c in enwik8_bytes()[1]] # validation data

    if name == 'wp-train':
        return [int(c) for c in enwik8_bytes()[0]]

    if name == 'wp-val':
        return [int(c) for c in enwik8_bytes()[1]]

    if name == 'wp-test':
        return [int(c) for c in enwik8_bytes()[2]]

