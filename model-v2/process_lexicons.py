"""
File: create-word-set.py
------------------------
Combines words from several sources into a document to create a
robust english language dictionary.
"""
import os
import re
import string
import pickle
from settings_v2 import COHA_DIRECTORY, COMMON_ENG_LEXICON
from settings_v2 import ENGLISH_LEXICON_PICKLED, COMMON_ENG_LEXICON_PICKLED

def main():
    store_english_lexicon()
    new_set = retrieve_english_lexicon()
    print("cat" in new_set)
    store_common_lexicon()
    new_set = retrieve_common_lexicon()
    print("cat" in new_set)


def store_english_lexicon():
    """
    Processes COHA and Google Files to generate a Hashset of as many recognizable English words as possible. Pickles the set and writes it to ENGLISH_LEXICON_PICKLED.
    """
    word_set = set()
    # adding all the COHA data
    for f in os.listdir(COHA_DIRECTORY):
        print(f"Processing COHA file {f}", end="\r")
        textfile = open(f"{COHA_DIRECTORY}{f}")
        for line in textfile:
            word = clean_text_v2(line.split("\t")[1])
            # we don't store 1 and 2 letter long words from COHA to be careful of false positives later on. Those come in from the Google data.
            if word.isalpha() and len(word) > 2:
                word_set.add(word)

    # adding all the Google data
    for line in open(COMMON_ENG_LEXICON):
        word_set.add(clean_text_v2(line))
    print("\nGoogle Data Processed.")

    with open(ENGLISH_LEXICON_PICKLED, "wb") as f:
        pickle.dump(word_set, f, pickle.HIGHEST_PROTOCOL)

    f.close()
    print("English lexicon written to file.")
    print(f"Total Words: {len(word_set)}")


def clean_text_v2(text):
    text = text.lower()
    text.strip("")

    new_text = ""
    for char in text:
        if char not in string.ascii_lowercase:
            continue
        new_text += char
    
    return new_text    


def retrieve_english_lexicon():
    """
    Retrieves the pickled English hashset from storage and returns it.
    """
    with open(ENGLISH_LEXICON_PICKLED, "rb") as f:
        return pickle.load(f)


def store_common_lexicon():
    """
    Processes the Google file to create a lexicon of common English words. Pickles the set and writes it to COMMON_ENG_LEXICON_PICKLED.
    """
    word_set = set()
    for line in open(COMMON_ENG_LEXICON):
        word_set.add(line.strip())
    
    with open(COMMON_ENG_LEXICON_PICKLED, "wb") as f:
        pickle.dump(word_set, f, pickle.HIGHEST_PROTOCOL)
    
    print("Common English lexicon written to file.")
    print(f"Total Words; {len(word_set)}")
    f.close()


def retrieve_common_lexicon():
    """
    Retrieves the pickled common English word hashset from storage and returns it.
    """
    with open(COMMON_ENG_LEXICON_PICKLED, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()