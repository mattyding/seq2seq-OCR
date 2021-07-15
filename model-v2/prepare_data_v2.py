"""
File: prepare_data_v2.py
--------------------
Generates training data with forced errors.

Stems training data.
"""
import os
import numpy as np
import string
from nltk.stem import PorterStemmer
from process_coha import clean_text_v2
from settings_v2 import ENGLISH_LEXICON, COMMON_ENG_LEXICON, COHA_DIRECTORY
from settings_v2 import DATA_PATH
from process_letter_sub import *


def main():
    write_training_data()


def write_training_data(replace=True):
    all_text = open(DATA_PATH, "w+" if (replace == True) else "a")

    # prepares all_text
    all_text.write(" \t \n")

    words = []
    for f in os.listdir(COHA_DIRECTORY):
        textfile = open(f"{COHA_DIRECTORY}{f}")
        print(f"Processing File: {f}", end="\r")
        for line in textfile:
            word = clean_text_v2(line.split("\t")[1])
            if word.isalpha():
                words.append(word)
    print("COHA Files Processed.")
    print(f"Total Number of Words: {len(words)}")

    # these retrieval functions are stored in "process_letter_sub.py" and unpickle stored data
    prob_sub_dict = retrieve_prob_sub_dict()
    error_prob_dict = retrieve_ocr_error_dict()

    for word in words:
        # every 3/20 characters is noisy
        all_text.write(simulate_ocr_noise(word, prob_sub_dict, error_prob_dict) + "\t" + word + "\n")
    
    print("Training data written.")
    all_text.close()


def simulate_ocr_noise(word, prob_sub_dict, error_prob_dict):
    """
    This function simulates OCR errors per the calculated error probabilities.
    """
    noisy_word = []
    
    for i in range(len(word)):
        currLetter = word[i]
        probError = error_prob_dict[currLetter]
        random = np.random.uniform(0, 1, 1)
        if (random < probError):
            noisy_word.append(get_random_letter_substitution(currLetter, prob_sub_dict))
            
        else:
            noisy_word.append(currLetter)
        
    noisy_word = "".join(noisy_word)
    return noisy_word if len(noisy_word) > 0 else word # in case short word + deletes letter


def get_random_letter_substitution(letter, sub_dict):
    """
    For any letter, takes the stored OCR substitution dict and pulls a random substitution from
    the dict of possible ones. Each substitution instance is equally likely, so the probability
    that a specific substitution occurs is equal to the frequency of that substitution in the
    source data. Therefore, more common substitutions are more likely to be returned.
    """
    assert(letter in string.ascii_lowercase)

    poss_subs = list(sub_dict[letter].keys())
    probs = list(sub_dict[letter].values())

    return np.random.choice(poss_subs, 1, replace=True, p=probs)[0]


if __name__ == "__main__":
    main()
