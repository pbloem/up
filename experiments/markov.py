from up.util import markov
from up.data import load_str

import fire

def go(name='toy', num_chars=1_00_000):

    print(f'# {name}')

    if name != 'wp':
        train, val, test = load_str(name=name, num_chars=num_chars), load_str(name=name, num_chars=num_chars), \
                       load_str(name=name, num_chars=num_chars)
    else:
        train, val, test = load_str(name='wp-train'), load_str(name='wp-val'), \
                       load_str(name='wp-test')

    # train, val, test = '110011', '110011', '00000'

    res = markov(train, val, test, max_order=5, verbose=True)

    print(res)
    print()
    print()

if __name__ == '__main__':

    go(name='dyck')
    go(name='toy')
    go(name='ndfa')
    go(name='wp')