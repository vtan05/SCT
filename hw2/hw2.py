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


def gen_bigram(filename):
    myFile = open(filename, 'r', encoding='utf-8')
    file = myFile.read()
    decoded = unidecode(file.lower())
    bigram = list(nltk.ngrams(decoded, 2))

    others = punctuation + digits + whitespace
    bigram_list = []
    for n in bigram:
        n = list(n)
        if n[0] in others:
            n[0] = '-'
        if n[1] in others:
            n[1] = '-'
        bigram_list.append(tuple(n))

    freq = nltk.ConditionalFreqDist(bigram_list) 

    prob_table = np.zeros((27, 27))
    for x in range(0, 27):
        for y in range(0, 27):
            prob_table[x][y] = freq[sorted(freq.conditions())[x]][sorted(freq.conditions())[y]] / freq.N()

    df = pd.DataFrame(prob_table, index=sorted(freq.conditions()), columns=sorted(freq.conditions()))
    fig = plt.figure()
    sns.heatmap(df, cmap='turbo', xticklabels=True, yticklabels=True)
    plt.title("Bigram Probabilities")
    plt.savefig(filename[:-4] + '_bigram.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    return freq, filename

def bigram_prob(freq, filename):
    prob = nltk.ConditionalProbDist(freq, nltk.MLEProbDist) 

    prob_table = np.zeros((27, 27))
    for x in range(0, 27):
        for y in range(0, 27):
            prob_table[x][y] = prob[sorted(prob.conditions())[y]].prob(sorted(prob.conditions())[x])

    df = pd.DataFrame(prob_table, index=sorted(prob.conditions()), columns=sorted(prob.conditions()))
    fig = plt.figure()
    sns.heatmap(df, cmap='turbo', xticklabels=True, yticklabels=True)
    plt.title("Bigram Conditional Probabilities P(x|y)")
    plt.savefig(filename[:-4] + '_condprob_xy.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    prob_table = np.zeros((27, 27))
    for x in range(0, 27):
        for y in range(0, 27):
            prob_table[x][y] = prob[sorted(prob.conditions())[x]].prob(sorted(prob.conditions())[y])

    df = pd.DataFrame(prob_table, index=sorted(prob.conditions()), columns=sorted(prob.conditions()))
    fig = plt.figure()
    sns.heatmap(df, cmap='turbo', xticklabels=True, yticklabels=True)
    plt.title("Bigram Conditional Probabilities P(y|x)")
    plt.savefig(filename[:-4] + '_condprob_yx.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    print('Processing: Noli Me Tangere (English ver.)')
    freq, filename = gen_bigram('noli_eng.txt')
    bigram_prob(freq, filename)

    #print('Processing: Noli Me Tangere (Tagalog ver.)')
    #freq, filename = gen_bigram('noli_tag.txt')
    #bigram_prob(freq, filename)
