from up.util import markov, markov_context
from up.data import load_str

import fire

def go(name='toy', num_chars=100_000, context=False, train_windows=10_000, test_windows=100_000):
    """

    :param name:
    :param num_chars:
    :param context: If false, trains a simple markov model in the training data, tunes the smoothing on val and evaluates
     in test. If true, trains a Markov model for every context window in test. We choose the smoothing parameter that
      gives the best results overall (not per window). This gives the model a slight advantage, but we accept this as
      it's used asa baseline.
    :return:
    """

    print(f'# {name}')

    if name != 'wp':
        train, val, test = load_str(name=name, num_chars=num_chars), load_str(name=name, num_chars=num_chars), \
                       load_str(name=name, num_chars=num_chars)
    else:
        train, val, test = load_str(name='wp-train'), load_str(name='wp-val'), \
                       load_str(name='wp-test')

    print('sample')
    print(train[:120])
    print()

    # train, val, test = '110011', '110011', '00000'

    if context:
        res = markov_context(train, test, max_order=5, train_windows=train_windows, test_windows=test_windows, verbose=True)
    else:
        res = markov(train, val, test, max_order=5, verbose=True)

    print(res)
    print()
    print()

if __name__ == '__main__':

    fire.Fire(go)
    # go(name='bits')
    # go(name='champ')
    # go(name='dyck')
    # go(name='toy')
    # go(name='ndfa')
    # go(name='wp')