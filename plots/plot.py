import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import json, glob, fire, os

from up.util import clean, ca
from up.data import MARKOV_CTX

import numpy as np

BASE = 4, 1
MULT = 4.0

def ablation_sources(dir='./ablation-sources/'):

    fig, subs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(BASE[0]*MULT, BASE[1]*MULT))

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
    for path in glob.glob(os.path.join(dir, '*.json')):
        with open(path, 'r') as file:
            datas.append(json.load(file))

    print(len(datas), 'files loaded')

    defcols = plt.rcParams['axes.prop_cycle'].by_key()['color'] # The default MPL colors
    colors = {
        'transformer' : defcols[0],
        'pointwise' : defcols[1],
        'uniform' : defcols[2],
        'ndfa' : defcols[3]
    }

    for d in datas:
        source = d['locals']['source'] if 'source' in d['locals'] else 'transformer'

        for name, res in d['vals'].items():
            x = res['instances']
            y = res['bits']

            axes[name].plot(x, y, linewidth=2, color=colors[source], label=source)

            axes[name].set_title(name)
            if name in ['toy', 'bits', 'wp']:
                axes[name].set_xlabel('instances seen')
            if name in ['champ', 'toy']:
                axes[name].set_ylabel('val loss (bits)')

    for name, values in MARKOV.items():
        for i, v in enumerate(values):
            axes[name].axhline(v, linestyle=':', color='gray', zorder=-1)

    subs[1][2].legend()

    plt.savefig('ablation-sources.png')
    plt.savefig('ablation-sources.pdf')

def scaling(dir='./scaling/'):

    fig, subs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(BASE[0]*MULT, BASE[1]*MULT))

    axes = {
            'champ': subs[0][0],
            'dyck': subs[0][1],
            'ndfa': subs[0][2],
            'toy':  subs[1][0],
            'bits': subs[1][1],
            'wp': subs[1][2]
    }

    ranges = {
        'champ': {'x' : (500_000, 7_000_000), 'y' : (3.2, 4.5)},
        'dyck': {'x' : (500_000, 7_000_000), 'y' : (1.4,2.5)},
        'ndfa': {'x' : (500_000, 7_000_000), 'y' : (3.5, 6)},
        'toy': {'x' : (500_000, 7_000_000), 'y' : (4.2, 6)},
        'bits': {'x' : (500_000, 7_000_000), 'y' : (1.1, 2.5)},
        'wp': {'x' : (500_000, 7_000_000), 'y' : (5, 7)}
    }

    axins = {}
    # Inset plots
    for name, axout in axes.items():

        axins[name] = axout.inset_axes([0.5, 0.5, 0.47, 0.47])

        axins[name].set_xscale('log')
        axins[name].set_yscale('log')
        axins[name].set_xlim(*ranges[name]['x'])
        axins[name].set_ylim(*ranges[name]['y'])

        clean(axins[name])
        rect, lines = axout.indicate_inset_zoom(axins[name], )
        for line in lines: # Turn off the connecting lines
            line.set(visible=False)

        axins[name].xaxis.set_major_formatter(NullFormatter())
        axins[name].xaxis.set_minor_formatter(NullFormatter())
        axins[name].yaxis.set_major_formatter(NullFormatter())
        axins[name].yaxis.set_minor_formatter(NullFormatter())

    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    # mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

    for ax in axes.values():
        clean(ax)

    datas = []
    for path in glob.glob(os.path.join(dir, '*.json')):
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

            axes[name].plot(x, y, linewidth=1, color=cmap[w2i[width]], label=f'{width}/{depth}')
            axins[name].plot(x, y, linewidth=1, color=cmap[w2i[width]], label=f'{width}/{depth}')

            axes[name].set_title(name)

            if name in ['toy', 'bits', 'wp']:
                axes[name].set_xlabel('instances seen')
            if name in ['champ', 'toy']:
                axes[name].set_ylabel('val loss (bits)')

    for name, values in MARKOV_CTX.items():

        opt = min(v for v in values)

        axes[name].axhline(opt, linestyle=':', color='gray', zorder=-1)
        axins[name].axhline(opt, linestyle=':', color='gray', zorder=-1)


    fig.tight_layout() # this needs to be before the adjust

    # Add the colorbar
    fig.subplots_adjust(right=0.87) # make some room
    cbar_ax = fig.add_axes([0.9, 0.1, 0.015, 0.8]) # new axes
    cbar_ax.set_title('model size (width/depth)', pad=12)
    mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.LogNorm(vmin=np.min(widths), vmax=np.max(widths)),
        cmap=bar)

    cbar = fig.colorbar(mappable=mappable, cax=cbar_ax)
    cbar.minorticks_off()
    # cbar.set_ticks(ticks=[], labels=[] )
    cbar.set_ticks(ticks=widths, labels=[f'{width}/{depth}' for width, depth in zip(widths, depths)] )

    # plt.tight_layout()
    plt.savefig('scaling.png')
    plt.savefig('scaling.pdf')


if __name__ == '__main__':
    fire.Fire()
