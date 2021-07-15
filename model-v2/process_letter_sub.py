"""
process_letter_sub.py
---------------------
This file contains several functions that prepare a dictionary of common OCR letter subtitutions.

The original source data is stored at LABELED_OCR_ERRORS; the source is given in the README.
"""
import numpy as np
import string
import pickle
from settings_v2 import LABELED_OCR_ERRORS, LETTER_SUBSTITUTIONS, LETTER_SUB_DICT

def main():
    store_sub_dict()
    print("RETRIEVAL")
    print_sub_dict(get_char_probability_dict())
    print_sub_dict(get_sub_probability_dict())

def store_sub_dict():
    """
    Converts a dict of letter/recorded OCR substitutions into pickle format.
    """    
    d = {}
    letterfile = open(LETTER_SUBSTITUTIONS)
    
    for line in letterfile:
        line = line.strip("\n")
        line = line.split("\t")
        orig_letter = line[0]
        bad_letter = line[1]

        if (orig_letter not in string.ascii_lowercase):
            print("BAD LETTER: ", orig_letter)
        elif (bad_letter not in string.ascii_lowercase):
            print("BAD LETTER: ", bad_letter)

        # if correct, continues onto the next pair of letters
        if (orig_letter == bad_letter):
            continue

        # else, appends it to the dict of possible substitutions
        if orig_letter not in d:
            d[orig_letter] = [bad_letter]
        else:
            d[orig_letter].append(bad_letter)
    
    with open(LETTER_SUB_DICT, "wb") as dict_file:
        pickle.dump(d, dict_file, pickle.HIGHEST_PROTOCOL)


def retrieve_sub_dict():
    """
    Retrieves the stored OCR substitution dict.
    """
    with open(LETTER_SUB_DICT, 'rb') as f:
        return pickle.load(f)


def get_random_letter_substitution(letter):
    """
    For any letter, takes the stored OCR substitution dict and pulls a random substitution from
    the dict of possible ones. Each substitution instance is equally likely, so the probability
    that a specific substitution occurs is equal to the frequency of that substitution in the
    source data. Therefore, more common substitutions are more likely to be returned.
    """
    assert(letter in string.ascii_lowercase)

    sub_dict = retrieve_sub_dict()
    return np.random.choice(np.array(sub_dict[letter]))


def store_ocr_error_dict():
    """
    Takes the stored OCR substitution data and combines it with COHA letter frequencies to find the
    probability that any specific letter encounters an OCR errors. Plots a chart of each letter and
    its associated probability. Saves the dictionary of letters and their probabilities as a binary file.
    """
    pass


def get_char_error_probability_dict():
    """
    Takes the stored OCR substitution dict and returns a new dictionary where each char is 
    mapped to the proportion of errors that they made up. Saves dictionary to binary file.
    """
    sub_dict = retrieve_sub_dict()

    total_count = {}
    for letter in sub_dict:
        total_count[letter] =  len(sub_dict[letter])
    
    total_subs = sum(total_count.values(), 0.0)
    return {letter: (freq / total_subs) for letter, freq in total_count.items()}


def get_char_probability_dict():
    """
    Counts up the frequency of letters stored in the sample COHA directory. 
    """
    
def get_sub_prob_dict():
    """
    Takes the stored OCR substituion dict and returns a new dictionary where each possible substitution
    for a character is mapped to its proportion of the substitions for that character (the probability
    that a randomly-chosen substitution for that character will be that one).
    """
    sub_dict = retrieve_sub_dict()

    new_prob_dict = {}

    for letter in sub_dict:
        sub_freq_dict = {}
        for poss_sub in sub_dict[letter]:
            if poss_sub not in sub_freq_dict:
                sub_freq_dict[poss_sub] = 1
            else:
                sub_freq_dict[poss_sub] += 1

        total_subs = sum(sub_freq_dict.values(), 0.0)
        new_prob_dict[letter] = {sub: (freq / total_subs) for sub, freq in sub_freq_dict.items()}
    
    return new_prob_dict


def print_sub_dict(sub_dict):
    """
    Convenience function to check the contents of any version of the OCR substitution dictionary.
    """
    for letter in string.ascii_lowercase:
        print(letter, sub_dict[letter], "\n")
        

def process_corrected_data():
    """
    Matches up OCR misreadings to their true letter and writes the letter-pairs to the LETTER_SUBSTITUTIONS file.
    """
    sourcefile = open(LABELED_OCR_ERRORS)
    newfile = open(LETTER_SUBSTITUTIONS, "w+")
    
    for line in sourcefile:
        line = line.split("\t")
        bad_word = line[0].lower()
        correct_word = line[1].lower()

        len_limit = min(len(bad_word), len(correct_word)) #if len(bad_word) < len(correct_word) else len(correct_word)
        i, j = 0, 1
        # i is the offset from the start of the word that is correct
        while (i < len_limit):
            if bad_word[i] != correct_word[i]:
                break
            i += 1
        # j is the offset from the end of the word that is correct
        while (j < len_limit):
            if bad_word[len(bad_word) - j] != correct_word[len(correct_word) - j]:
                break
            j += 1

        # isolates the part of the word that mismatches
        bad_part, correct_part = "", ""
        for k in range(i, len_limit - j + 1):
            try:
                bad_part += bad_word[k]
                correct_part += correct_word[k]
            except:
                pass

        for i in range(len(bad_part)):
            if any(char not in string.ascii_lowercase for char in [correct_part[i], bad_part[i]]):
                continue
            newfile.write(f"{correct_part[i]}\t{bad_part[i]}\n")
        
    sourcefile.close()
    newfile.close()


if __name__ == "__main__":
    main()