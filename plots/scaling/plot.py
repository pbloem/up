import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import json, glob, fire

from up.util import clean, ca

import numpy as np

MARKOV = {
    'wp' : [5.106629000536541, 3.920734110007795, 3.0992338340343406, 2.5360331955140865, 2.2364564738213955, 2.1894437828412445],
    'dyck' : [1.37402480190898, 1.1336195970119751, 1.133641671511376, 0.9372411063969551, 0.9372486322251692, 0.8904998264766827],
    'ndfa' : [3.4569746915375603, 0.6658862666099562, 0.2942196511608079, 0.2942189625751208, 0.2942892796818609, 0.29438668804011764],
    'toy' : [4.196220178391961, 2.2355184935857064, 0.9741857673348161, 0.7252455261062091, 0.6389911594450575, 0.6320596343654071],
    'bits': [1.1733818782355905, 1.1691583755645572, 1.166750905432737, 1.164872046306139, 1.1631674006093808, 1.1381807911247805],
    'champ': [4.020301090644664, 4.182040925227881, 4.475985937327467, 4.309589578934016, 4.104379828690582, 4.090988560452365]
}

def go():

    base = 16, 4
    mult = 1.3
    fig, subs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(base[0]*mult, base[1]*mult))

    axes = {
            'champ': subs[0][0],
            'dyck': subs[0][1],
            'ndfa': subs[0][2],
            'toy':  subs[1][0],
            'bits': subs[1][1],
            'wp': subs[1][2]
    }

    for ax in axes.values():
        clean(ax)

    datas = []
    for path in glob.glob('*.json'):
        with open(path, 'r') as file:
            datas.append(json.load(file))

    print(len(datas), 'files loaded')
    widths = np.asarray(sorted(d['locals']['width'] for d in datas))
    depths = np.asarray(sorted(d['locals']['depth'] for d in datas))
    lnwidths = np.log2(widths)
    lnwidths = (lnwidths - np.min(lnwidths)) / (np.max(lnwidths) - np.min(lnwidths))

    w2i = {w:i for i, w in enumerate(widths)}

    bar = plt.cm.cividis_r
    cmap = bar(lnwidths)

    for d in datas:
        width = d['locals']['width']
        depth = d['locals']['depth']
        for name, res in d['vals'].items():
            x = res['instances']
            y = res['bits']

            axes[name].plot(x, y, linewidth=3, color=cmap[w2i[width]], label=f'{width}/{depth}')

            axes[name].set_title(name)
            if name in ['toy', 'bits', 'wp']:
                axes[name].set_xlabel('instances seen')
            if name in ['champ', 'toy']:
                axes[name].set_ylabel('val loss (bits)')

    for name, values in MARKOV.items():
        for i, v in enumerate(values):
            axes[name].axhline(v, linestyle=':', color='gray', zorder=-1)

    # Add the colorbar

    fig.subplots_adjust(right=0.87) # make some room
    cbar_ax = fig.add_axes([0.9, 0.1, 0.015, 0.8]) # new axes
    # ca(cbar_ax)

    mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.LogNorm(vmin=np.min(widths), vmax=np.max(widths)),
        cmap=bar)

    cbar = fig.colorbar(mappable=mappable, cax=cbar_ax)
    cbar.minorticks_off()
    # cbar.set_ticks(ticks=[], labels=[] )
    cbar.set_ticks(ticks=widths, labels=[f'{width}/{depth}' for width, depth in zip(widths, depths)] )

    plt.savefig('scaling.png')
    plt.savefig('scaling.pdf')


if __name__ == '__main__':
    fire.Fire(go)
