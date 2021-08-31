"""
process_letter_sub.py
---------------------
This file contains several functions that prepare a dictionary of common OCR letter subtitutions.

The original source data is stored at LABELED_OCR_ERRORS; the source is given in the README.
"""
import numpy as np
import matplotlib.pylab as plt
import os
import string
import pickle
from settings import LABELED_OCR_ERRORS, LETTER_SUBSTITUTIONS, LETTER_PROB_FIG
from settings import NUM_LETTER_SUB_DICT, PROB_LETTER_SUB_DICT, ERROR_PROB_DICT


def rerun_data_processing():
    """
    Reruns all scripts. Takes the original source data, processes it, re-calculates probabilities,
    and re-saves all models.
    """
    print("Re-evaluating functions on source data.")
    process_corrected_data()
    store_num_sub_dict()
    store_prob_sub_dict()
    store_ocr_error_dict()
    graph_error_prob()
    print(f"Program finished. Graph of error probabilities written to: \n{LETTER_PROB_FIG}\n")


def print_all_dictionaries():
    """
    Prints all stored substitution/prediction dictionaries.
    """
    print("SUB DICTIONARY (COUNTS): ")
    print_saved_dict(retrieve_num_sub_dict())
    print("SUB DICTIONARY (PROBABILITIES): ")
    print_saved_dict(retrieve_prob_sub_dict())
    print("OCR ERROR DICTIONARY: ")
    print_saved_dict(retrieve_ocr_error_dict())


def store_num_sub_dict():
    """
    Converts a dict of letter/recorded OCR substitutions into pickle format. The value of each 
    letter is a dict storing possible substitutions and the number of times they were counted in
    the original dataset.
    """    
    d = {}
    letterfile = open(LETTER_SUBSTITUTIONS)
    
    for line in letterfile:
        line = line.strip("\n")
        line = line.split("\t")
        orig_letter = line[0]
        bad_letter = line[1]
        multiplier = int(line[2])  # number of times this substitution occurred

        # if correct, continues onto the next pair of letters
        if (orig_letter == bad_letter):
            continue

        # else, appends it to the dict of possible substitutions
        if orig_letter not in d:
            d[orig_letter] = {bad_letter : multiplier}
        else:
            if bad_letter in d[orig_letter]:
                d[orig_letter][bad_letter] += multiplier
            else:
                d[orig_letter][bad_letter] = multiplier
    
    with open(NUM_LETTER_SUB_DICT, "wb") as f:
        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def retrieve_num_sub_dict():
    """
    Retrieves the stored OCR substitution dict.
    """
    with open(NUM_LETTER_SUB_DICT, 'rb') as f:
        return pickle.load(f)


def store_prob_sub_dict():
    """
    Takes the raw count of letter substitutions and converts each one into a probability. The new stored
    value reflects the probability that a random substitution will be that specific character.
    """
    num_sub_dict = retrieve_num_sub_dict()
    prob_sub_dict = {}

    for letter in num_sub_dict:
        total_poss = sum(num_sub_dict[letter].values())
        prob_sub_dict[letter] = {k : (v / total_poss) for k, v in num_sub_dict[letter].items()}
    
    with open(PROB_LETTER_SUB_DICT, "wb") as f:
        pickle.dump(prob_sub_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def retrieve_prob_sub_dict():
    """
    Retrieves the stored OCR substitution probability dict.
    """
    with open(PROB_LETTER_SUB_DICT, 'rb') as f:
        return pickle.load(f)


def store_ocr_error_dict():
    """
    Takes the labeled OCR data and calculates the proportion of each letter that have OCR errors. Plots
    a chart of each letter and its associated probability. Saves the dictionary of letters and their 
    probabilities as a binary file.
    """
    char_error_prob_dict = get_char_error_probability_dict()
    char_prob_dict = get_char_probability_dict()

    ocr_error_dict = {}

    for letter in string.ascii_lowercase:
        ocr_error_dict[letter] = (char_error_prob_dict[letter] / char_prob_dict[letter])
    
    with open(ERROR_PROB_DICT, "wb") as dict_file:
        pickle.dump(ocr_error_dict, dict_file, pickle.HIGHEST_PROTOCOL)


def retrieve_ocr_error_dict():
    with open(ERROR_PROB_DICT, 'rb') as f:
        return pickle.load(f)


def get_char_error_probability_dict():
    """
    Takes the stored OCR substitution dict and returns a new dictionary where each char is 
    mapped to the number of OCR errors it has recorded.
    """
    num_sub_dict = retrieve_num_sub_dict()

    total_count = {}
    for letter in num_sub_dict:
        total_count[letter] =  sum(num_sub_dict[letter].values())

    return total_count


def get_char_probability_dict():
    """
    Takes the labeled OCR data and counts the number of times each character was present
    in the corrected word. Returns a dict mapping each character to its count.
    """
    char_prob_dict = {}
    for letter in string.ascii_lowercase:
        char_prob_dict[letter] = 0
    
    sourcefile = open(LABELED_OCR_ERRORS)

    for line in sourcefile:
        word = line.split("\t")[1]  # corrected word
        multiplier = int(line.split("\t")[2]) # number of times word appears
        for char in word:
            if char not in string.ascii_lowercase:
                continue
            char_prob_dict[char] += multiplier
    
    return char_prob_dict


def graph_error_prob():
    """
    Function to graph the OCR error probabilities for each letter.
    """
    err_prob_dict = retrieve_ocr_error_dict()

    lists = sorted(err_prob_dict.items()) # sorted by key, return a list of tuples

    x, y = zip(*lists) # unpack a list of pairs into two tuples

    fig = plt.figure(linewidth=5, edgecolor="#04253a")

    plt.bar(x, y)
    plt.ylabel("Probability of Error")
    plt.title("OCR Error Probability per Letter")
    plt.gca().set(ylim=(0, 1))
    fig.savefig(
    LETTER_PROB_FIG,
    edgecolor=fig.get_edgecolor())


def print_saved_dict(d):
    """
    Convenience function to check the contents of any version of the OCR substitution/probability dictionary.
    """
    for letter in string.ascii_lowercase:
        print(letter, d[letter], "\n")
        

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
            # discards pair if the correct character is not a letter
            if correct_part[i] not in string.ascii_lowercase:
                continue
            # in pre-processing the text we want to infer, we remove non-alphabetical 
            # characters. So if misread char is non-alphabetic, we replace it with an empty string
            bad_char = "" if bad_part[i] not in string.ascii_lowercase else bad_part[i]
            
            newfile.write(f"{correct_part[i]}\t{bad_char}\t{line[2]}")

    sourcefile.close()
    newfile.close()


if __name__ == "__main__":
    rerun_data_processing()
