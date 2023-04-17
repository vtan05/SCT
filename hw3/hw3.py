# Scientific Concepts and Thinking
# HW 3: Codes and Bayes' Theorem
# Submitted by Vanessa Tan 20225640

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from unidecode import unidecode
from string import punctuation, digits, whitespace, ascii_uppercase


count = lambda l1,l2: sum([1 for x in l1 if x in l2])

def mono_prob(decoded, alphabet):

    letter_freq = []
    for letter in alphabet:
        freq = round(decoded.count(letter) / len(decoded), 4)
        letter_freq.append(freq)
        #print(letter + ": " + str(freq))

    others = punctuation + digits + whitespace
    others_freq = round(count(decoded, others) / len(decoded), 4)
    letter_freq.append(others_freq)
    #print('_ :' + str(others_freq))

    return letter_freq


def gen_cipher(text, x, alphabet):

    shift = alphabet[x:] + alphabet[:x]
    table = text.maketrans(alphabet, shift)
    
    return text.translate(table)

def bayes(prob, decoded):

    ###### Proving Bayes' Theorem ######

    p_ab = prob['V'].prob('Q')
    p_b = decoded.count('V') / len(decoded)
    p_ba = prob['Q'].prob('V')
    p_a = decoded.count('Q') / len(decoded)

    print("Proving Bayes' Theorem")
    print(round((p_ab * p_b), 4))
    print(round((p_ba * p_a), 4))


def bigram_prob(freq, decoded):

    ###### Compute bigram conditional probabilities ######

    prob = nltk.ConditionalProbDist(freq, nltk.MLEProbDist) 

    prob_table = np.zeros((23, 23))
    for x in range(0, 23):
        for y in range(0, 23):
            prob_table[x][y] = prob[sorted(prob.conditions())[y]].prob(sorted(prob.conditions())[x])

    df = pd.DataFrame(prob_table, index=sorted(prob.conditions()), columns=sorted(prob.conditions()))
    fig = plt.figure()
    sns.heatmap(df, cmap='turbo', xticklabels=True, yticklabels=True)
    plt.title("Bigram Conditional Probabilities P(x|y) of Enciphered Text")
    plt.savefig('condprob_xy.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    prob_table = np.zeros((23, 23))
    for x in range(0, 23):
        for y in range(0, 23):
            prob_table[x][y] = prob[sorted(prob.conditions())[x]].prob(sorted(prob.conditions())[y])

    df = pd.DataFrame(prob_table, index=sorted(prob.conditions()), columns=sorted(prob.conditions()))
    fig = plt.figure()
    sns.heatmap(df, cmap='turbo', xticklabels=True, yticklabels=True)
    plt.title("Bigram Conditional Probabilities P(y|x) of Enciphered Text")
    plt.savefig('condprob_yx.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    bayes(prob, decoded)

def gen_bigram(decoded):

    ###### Generate bigram probabilities ######
    bigram = list(nltk.ngrams(decoded, 2))

    others = punctuation + digits + whitespace
    bigram_list = []
    i = 0
    for n in bigram:
        n = list(n)
        if n[0] in others:
            n[0] = '-'
        if n[1] in others:
            n[1] = '-'
        bigram_list.append(tuple(n))
        i += 1

    freq = nltk.ConditionalFreqDist(bigram_list) 

    prob_table = np.zeros((23, 23))
    for x in range(0, 23):
        for y in range(0, 23):
            prob_table[x][y] = freq[sorted(freq.conditions())[x]][sorted(freq.conditions())[y]] / freq.N()

    df = pd.DataFrame(prob_table, index=sorted(freq.conditions()), columns=sorted(freq.conditions()))
    fig = plt.figure()
    sns.heatmap(df, cmap='turbo', xticklabels=True, yticklabels=True)
    plt.title("Bigram Probabilities of Enciphered Text")
    plt.savefig('bigram.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    bigram_prob(freq, decoded)

if __name__ == "__main__":
    remove = ['J', 'U', 'W']
    alphabet = list(ascii_uppercase)

    for letter in remove:
        alphabet.remove(letter)
    alphabet = ''.join([str(elem) for elem in alphabet])

    ##### Generate all 23 possible ciphers #####
    #text = 'ROMAMOPPVGNATE'
    #for i in range(23):
    #    #print(i)
    #    cipher = gen_cipher(text, i, alphabet)
    #    #print(cipher)

    ##### Generate monogram probabilities of all letters #####
    file = 'bellogallicoincaps.txt'
    myFile = open(file, 'r', encoding='utf-8')
    file = myFile.read()
    decoded = unidecode(file.upper())
    
    #freq = mono_prob(decoded, alphabet)

    ##### Generate monogram probabilities of all letters (X = 5) #####
    #cipher = gen_cipher(decoded, 5, alphabet)
    #freq = mono_prob(cipher, alphabet)

    ##### Generate monogram probabilities of all letters from the one-time pad #####
    otp_file = 'xotp.txt'
    myOtp = open(otp_file, 'r', encoding='utf-8')
    otp_file = myOtp.read()
    otps = list(otp_file.split("\n"))
    
    # monoprobs = []
    # for otp in range(len(otps)):
    #     cipher = gen_cipher(decoded, int(otps[otp]), alphabet)
    #     freq = mono_prob(cipher, alphabet)
    #     monoprobs.append(freq)
    # monoprobs = np.array(monoprobs)

    # for i in range(0,24):
    #     if i < 23:
    #         print(alphabet[i] + ": " + str(np.mean(monoprobs[:,i])))
    #     else:
    #         print("_: " + str(np.mean(monoprobs[:,i])))

    ##### Generate bigram probabilities of original text and Bayes' Theorem #####
    gen_bigram(decoded)

    ##### Generate bigram probabilities of enciphered text #####
    # cipher = gen_cipher(decoded, int(otps[0]), alphabet)
    # gen_bigram(cipher)
    
