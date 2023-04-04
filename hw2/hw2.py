# Scientific Concepts and Thinking
# HW 2: Probabilities
# Submitted by Vanessa Tan 20225640

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from unidecode import unidecode
from string import punctuation, digits, whitespace, ascii_lowercase


def drawPlot(propTable, xTickLabel, yTickLabel, title, size_scale, saveName):
    fig, ax = plt.subplots()
    x_size = np.size(propTable, 0)
    y_size = np.size(propTable, 1)

    x, y = np.meshgrid(np.arange(x_size), np.arange(y_size))

    im = ax.scatter(
        x=y,
        y=x,
        s=propTable.transpose() * size_scale,
        c=propTable.transpose(),
        cmap='RdYlGn',
        marker='s'
    )

    ax.set(xticks=np.arange(y_size), yticks=np.arange(x_size),
           xticklabels=yTickLabel, yticklabels=xTickLabel)
    ax.set_xticks(np.arange(y_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(x_size + 1) - 0.5, minor=True)
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.grid(which='minor')
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-0.5, (y_size-1) + 0.5])
    ax.set_ylim([-0.5, (x_size-1) + 0.5])
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_title(title, loc='center')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(saveName + '.png', dpi=300, bbox_inches='tight')
    plt.show()

def gen_bigram(file):
    myFile = open(file, 'r', encoding='utf-8')
    file = myFile.read()
    decoded = unidecode(file.lower())

    bigram = list(nltk.ngrams(decoded, 2))

    others = punctuation + digits + whitespace
    bigram_list = []
    for ele in bigram:
        ele = list(ele)
        if ele[0] in others:
            ele[0] = '-'
        if ele[1] in others:
            ele[1] = '-'
        bigram_list.append(tuple(ele))

    freq = nltk.ConditionalFreqDist(bigram_list)     

    conditions = list(ascii_lowercase)
    conditions.append('-')

    pd_bigram_table = np.zeros((27, 27))
    for x in range(0, 27):
        for y in range(0, 27):
            pd_bigram_table[x][y] = freq[conditions[x]][conditions[y]] / freq.N()

    bigram_df = pd.DataFrame(pd_bigram_table, index=conditions, columns=conditions)

    # drawPlot(pd_bigram_table, conditions, conditions,
    #      "Probability distribution P(X=x, Y=y) for ordered pair XY", 1000,
    #      "fig_bigram")
    sns.heatmap(bigram_df, cmap="crest")
    plt.show()

if __name__ == "__main__":
    print('\nNoli Me Tangere (English ver.)')
    gen_bigram('noli_eng.txt')
    # print('\nNoli Me Tangere (Tagalog ver.)')
    # countAlphabet('noli_tag.txt')
