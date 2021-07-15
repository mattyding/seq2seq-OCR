"""
File: prepare_data_v2.py
--------------------
Generates training data with forced errors.

Stems training data.
"""
import os
import numpy as np
import string
import pickle
from nltk.stem import PorterStemmer
from process_coha import clean_text_v2
from settings_v2 import ENGLISH_LEXICON, COMMON_ENG_LEXICON, COHA_DIRECTORY
from settings_v2 import DATA_PATH
from process_letter_sub import *


OBSERVERED_ERROR_FREQ = 0.85  # proportion of letters that are erroneous
OBSERVERED_REPLACE_FREQ = 0.8 # propoprtion of letter errors that are replacement instead of ommission


def main():
    write_training_data(False)

def write_training_data(replace=True):
    all_text = open(DATA_PATH, "w+" if (replace == True) else "a")

    # prepares all_text
    all_text.write(" \t \n")

    sub_dict = retrieve_sub_dict()

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
        all_text.write(noise_maker(word, OBSERVERED_ERROR_FREQ, sub_dict) + "\t" + word + "\n")

    all_text.close()


def noise_maker(word, threshold, letterdict):
    letters = [c for c in string.ascii_lowercase]
    noisy_word= []
    i = 0
    while i < len(word):
        random = np.random.uniform(0,1,1)
        if random < threshold:
            noisy_word.append(word[i])  # adds the letter normally
        else:

            new_random = np.random.uniform(0,1,1)
            if new_random < OBSERVERED_REPLACE_FREQ:
                # substitutes in letter using OCR error dict
                if word[i] in letterdict:
                    noisy_word.append(str(np.random.choice(letterdict[word[i]])))
                else:
                    noisy_word.append(np.random.choice([c for c in string.ascii_lowercase]))
            else:
                # does not type letter
                pass     
        i += 1
    noisy_word = "".join(noisy_word)
    return noisy_word if len(noisy_word) > 0 else word # in case short word + deletes letter

if __name__ == "__main__":
    main()
