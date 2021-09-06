"""
File: prepare_data_v2.py
--------------------
Generates training data with forced errors.
"""
import os
import numpy as np
import string
from settings import COHA_DIRECTORY, DATA_PATH, MAX_SEQ_LENGTH, MIN_SEQ_LENGTH
from process_letter_sub import retrieve_prob_sub_dict, retrieve_ocr_error_dict
from general_util import print_statusline, clean_text_no_spaces


def main():
    write_training_data()


def write_training_data():
    # these retrieval functions are stored in "process_letter_sub.py" and unpickle stored data
    prob_sub_dict = retrieve_prob_sub_dict()
    error_prob_dict = retrieve_ocr_error_dict()

    word_pairs = []
    for f in os.listdir(COHA_DIRECTORY):
        textfile = open(f"{COHA_DIRECTORY}{f}")
        print_statusline(f"Processing File: {f}")
        for line in textfile:
            word = line.split("\t")[1]
            word = clean_text_no_spaces(word)
            # model is only evaluated on words >=3 characters
            if (len(word) < MIN_SEQ_LENGTH) or len(word) > MAX_SEQ_LENGTH or not word.isalpha():
                continue
            noisy_word = word
            while (noisy_word == word):
                # each character is evaluated against its probability of error
                noisy_word = simulate_ocr_noise(word, prob_sub_dict, error_prob_dict)
            word_pairs.append(noisy_word + "\t" + word + "\n")

    print_statusline("COHA Files Processed.\n")
    print(f"Total Number of Words: {len(word_pairs)}")
    print("Writing Training Data to File.")

    with open(DATA_PATH, "w") as training_file:
        for pair in word_pairs:
            training_file.write(pair)
        training_file.close()

    print("All Training Data Written.")


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
            # get_random_letter_substituion() can sometimes return nothing (pseudo-deletion)
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
