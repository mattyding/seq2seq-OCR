"""
File: process_lexicons.py
------------------------
Combines words from several sources into a document to create a robust english language lexicon.
"""
import os
import pickle
from general_util import print_statusline, clean_text_no_spaces
from settings import COHA_DIRECTORY, COMMON_ENG_LEXICON
from settings import ENGLISH_LEXICON_PKL, COMMON_ENG_LEXICON_PKL


def main():
    store_english_lexicon()
    store_common_lexicon()

def store_english_lexicon(override=True):
    """
    Processes COHA and Google Files to generate a Hashset of as many recognizable English words as possible. Writes the set to ENGLISH_LEXICON_PKL
    """
    word_set = set()
    # adding all the COHA data
    for f in os.listdir(COHA_DIRECTORY):
        print_statusline(f"Processing COHA file: {f}")
        textfile = open(f"{COHA_DIRECTORY}{f}")
        for line in textfile:
            # removes hypenated words
            if '-' in line:
                continue
            clean_word = clean_text_no_spaces(line.split("\t")[1])
            # we don't store 1 and 2 letter long words from COHA to be careful of false positives later on. Those come in from the Google data.
            if len(clean_word) > 3:
                word_set.add(clean_word)
        textfile.close()

    # adds back in very common 3-letter words
    with open(COMMON_ENG_LEXICON, 'r') as f:
        for line in f:
            word_set.add(clean_text_no_spaces(line))
        f.close()
    
    with open(ENGLISH_LEXICON_PKL, mode='wb' if override else 'xb') as f:
        pickle.dump(word_set, f, pickle.HIGHEST_PROTOCOL)

    f.close()
    print("English lexicon written to file.")
    print(f"Total Words: {len(word_set)}")

def store_common_lexicon(override=True):
    """
    Processes the Google file to create a lexicon of common English words. Writes the set to COMMON_ENG_LEXICON_PKL.
    """
    word_set = set()
    for line in open(COMMON_ENG_LEXICON):
        word_set.add(clean_text_no_spaces(line))
    
    with open(COMMON_ENG_LEXICON_PKL, mode='wb' if override else 'xb') as f:
        pickle.dump(word_set, f, pickle.HIGHEST_PROTOCOL)
    
    print("Common English lexicon written to file.")
    print(f"Total Words; {len(word_set)}")
    f.close()


if __name__ == "__main__":
    main()