import os
import glob
import re
import pandas as pd
import numpy as np
import textdistance
import string
from process_coha import clean_text_v2
from settings import TEXT_DIRECTORY, COMB_DIRECTORY

ENGLISH_LEXICON = "./english-words.txt"
COMMON_ENG_LEXICON = "./google-10000-english-no-swears.txt"
COHA_DIRECTORY = "./COHA-sample-data/"


def main():
    all_text = open(TEXT_DIRECTORY + "ALL_TEXT.txt", "w+")

    # prepares all_text
    all_text.write(" \t \n")

    words = []
    for f in os.listdir(COHA_DIRECTORY):
        textfile = open(f"./{COHA_DIRECTORY}/{f}")
        print(f"Processing File: {f}")
        for line in textfile:
            word = clean_text_v2(line.split("\t")[1])
            if word.isalpha():
                words.append(word)
        print(f"{f} file processed.")

    for word in words:
        all_text.write(noise_maker(word, 0.95) + "\t" + word + "\n")


def noise_maker(sentence, threshold):
    letters = [c for c in string.ascii_lowercase]
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    continue
                else:
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            elif new_random < 0.33:
                random_letter = np.random.choice([c for c in string.ascii_lowercase])
                noisy_sentence.append(random_letter)
                noisy_sentence.append(sentence[i])
            else:
                pass     
        i += 1
    return "".join(noisy_sentence)

if __name__ == "__main__":
    main()