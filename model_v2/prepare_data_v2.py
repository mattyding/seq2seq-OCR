"""
File: prepare_data_v2.py
--------------------
Generates training data with forced errors.

Stems training data.
"""
import os
import glob
import re
import pandas as pd
import numpy as np
import textdistance
import string
from nltk.stem import PorterStemmer
from process_coha import clean_text_v2
from settings_v2 import ENGLISH_LEXICON, COMMON_ENG_LEXICON, COHA_DIRECTORY, LETTER_SUB_DIRECTORY
from settings_v2 import DATA_PATH


def main():
    all_text = open(DATA_PATH, "w+")

    # prepares all_text
    all_text.write(" \t \n")

    """ prepares noisy OCR patterns """
    d = {}
    letterfile = open(LETTER_SUB_DIRECTORY)
    for line in letterfile:
        line = line.strip("\n")
        line = line.split("\t")
        orig_letter = line[0]
        bad_letter = line[1]
        if orig_letter not in d:
            d[orig_letter] = np.full(1, bad_letter)
        else:
            d[orig_letter] = np.append(d[orig_letter], bad_letter)
    """
    if orig_letter not in d:
        d[orig_letter] = {bad_letter: 1}
    elif bad_letter not in d[orig_letter]:
        d[orig_letter][bad_letter] = 1
    else:
        d[orig_letter][bad_letter] += 1
    for item in d:
        print(item, d[item])
    return
    """

    ps = PorterStemmer()
    words = []
    for f in os.listdir(COHA_DIRECTORY):
        textfile = open(f"{COHA_DIRECTORY}{f}")
        print(f"Processing File: {f}", end="\r")
        for line in textfile:
            word = clean_text_v2(line.split("\t")[1])
            if word.isalpha():
                words.append(ps.stem(word))
        print(f"{f} file processed.")
    print("COHA Files Processed")

    for word in words:
        # every 3/20 characters is noisy
        all_text.write(noise_maker(word, 0.85, d) + "\t" + word + "\n")


def noise_maker(sentence, threshold, letterdict):
    letters = [c for c in string.ascii_lowercase]
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            if new_random < 0.85: #substitutes in letter using OCR error dict
                if sentence[i] in letterdict:
                    noisy_sentence.append(str(np.random.choice(letterdict[sentence[i]])))
                else:
                    noisy_sentence.append(np.random.choice([c for c in string.ascii_lowercase]))
                    """
                    elif new_random < 0.33: # adds in random letter
                        random_letter = np.random.choice([c for c in string.ascii_lowercase])
                        noisy_sentence.append(random_letter)
                        noisy_sentence.append(sentence[i])
                    """
            else: # does not type letter
                pass     
        i += 1
    noisy_word = "".join(noisy_sentence)
    return noisy_word if len(noisy_word) > 0 else sentence  # in case short word + deletes letter

if __name__ == "__main__":
    main()