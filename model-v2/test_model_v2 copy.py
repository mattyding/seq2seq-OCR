"""
test_model_v2
-------------
THIS COPY DOESNT HAVE WORD FREQUENCY GRAPHS AND IMPLEMENTS A NEW THING TO CHECK FOR COMPOUND WOWRDS


This program evaluates the trained model and uses it to predict text.

To use, put a folder of text files in the directory specified by DOC_DIRECTORY
see settings.py for more details and to change the location where the program tests.
"""
import os
import pathlib
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from process_lexicons import clean_text_v2, retrieve_english_lexicon, retrieve_common_lexicon
from settings_v2 import DATA_PATH, LATENT_DIM, NUM_SAMPLES, BREAK_CHAR
from settings_v2 import SAVED_MODEL, FREQ_DIRECTORY, DOC_DIRECTORY, PREDICTED_DIRECTORY


def evaluate_model():        
    """ Retrieves English Hashsets """
    english_words = retrieve_english_lexicon()  # large set containing many English words
    common_lexicon = retrieve_common_lexicon()  # only commonly-used words


    """ Preparing Model """
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines[: min(NUM_SAMPLES, len(lines) - 1)]:
        # tab is our break char and newline is end seq char
        input_text, target_text = line.split(BREAK_CHAR)
        target_text = "\t" + target_text + "\n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    model = keras.models.load_model(SAVED_MODEL)

    encoder_inputs = model.input[0]  # input 1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm 1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input 2
    decoder_state_input_h = keras.Input(shape=(LATENT_DIM,), name="input_3")
    decoder_state_input_c = keras.Input(shape=(LATENT_DIM,), name="input_4")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index["\t"]] = 1.0

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            states_value = [h, c]
        return decoded_sentence

    """ Testing """

    for folderpath in os.listdir(DOC_DIRECTORY):
        folder = folderpath.split("/")[-1]
        print(f"\nCURRENT FOLDER: {folder}")
        if folder == ".DS_Store":
            continue
        pathlib.Path(PREDICTED_DIRECTORY + folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(FREQ_DIRECTORY + folder).mkdir(parents=True, exist_ok=True)
        for f in os.listdir(DOC_DIRECTORY + folder + "/"):
            if f == ".DS_Store":
                continue
            # This checks if it's already been processed. If you want to re-run, delete the predicted text file
            if ("P_" + f) in os.listdir(PREDICTED_DIRECTORY + folder + "/"):
                print(f"File Already Exists: {f}")
                continue
            else:
                doc_text = []
                textfile = open(f"{DOC_DIRECTORY + folder}/{f}")
                try:
                    for line in textfile:
                        line = line.replace("\n", " ")
                        line = line.replace("\t", "")
                        line = line.replace("- ", "")
                        for word in line.split(" "):
                            word = clean_text_v2(word)
                            if len(word) > 0:
                                doc_text.append(word)
                except:
                    print(f"Unicode Error: {f}")
                    continue

                num_encoder_tokens = encoder_inputs.shape[2]

                couching = False # couching bool tracks if the next word has already been included

                translated_doc = []
                for i in range(len(doc_text)):
                    # next word has been included; skips it
                    if couching:
                        couching = False
                        continue

                    word = doc_text[i]

                    # if next word is recognizable, does not alter it
                    if word in english_words:
                        translated_doc.append(word)
                    # else if the next two entries make a word, couches them
                    elif ((i != len(doc_text) - 1) and ((word + doc_text[i+1]) in english_words)):
                        translated_doc.append(word + doc_text[i+1])
                        couching = True
                    else:
                        # else if the word can be split up into valid words (i.e., missing spaces between words)
                        split_word = check_compound(word, common_lexicon)
                        if (len(split_word) != len(word)):
                            translated_doc.append(split_word)
                        # if those pre-checks don't pass, feeds the word into the seq2seq model
                        else:
                            try:
                                word_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
                                for t, char in enumerate(word):
                                    word_data[0, t, input_token_index[char]] = 1.0
                                word_data[0, t + 1 :, input_token_index[" "]] = 1.0
                                decoded_word = ""
                                input_seq = word_data[0:1]
                                decoded_word += decode_sequence(input_seq)
                                decoded_word = decoded_word.strip("\n")
                                # only replaces original if model outputs a valid English word
                                if decoded_word in english_words:
                                    translated_doc.append(decoded_word)
                                else:
                                    translated_doc.append(word)
                            except:
                                print(f"Can't read word in {f}: {word}")
                                translated_doc.append("...")


                predicted_doc = open(PREDICTED_DIRECTORY + folder + "/P_" + f, "w+")
                predicted_doc.write(" ".join(translated_doc))
                predicted_doc.close()
                print("File Processed: " + f)


def check_compound(word, common_lexicon):
    # list containing all of the recursively found splittings
    found_splits = []

    check_compound_recursive([word], common_lexicon, found_splits)

    # no splits found
    if len(found_splits) == 0:
        return word
    
    # returns the splitting with fewest total splits (largest word-sections)
    fewest_splits = min(found_splits, key=len)

    # check to make sure it didn't find 2-word splits throughout (rejects < 3 letter average)
    if len(word) / len(fewest_splits) < 3:
        return word

    return " ".join(fewest_splits)


def check_compound_recursive(word_list, english_set, found_splits):
    """
    Recursive function to check if a word is a compound word (possibly caused by missing spaces between
    two or more words. Each word in the division must be at least 4 letters long to safeguard against
    false positives. If a proper seperation of the word is found, it is returned.
    """
    BUFFER = 2

    # Base Case: the provided word can be compounded
    if all(word in english_set for word in word_list):
        found_splits.append(word_list)
    # Recursive step, iterates through the chars of last item until the first part makes a word
    for i in range(BUFFER, len(word_list[-1]) - (BUFFER - 1)):  # BUFFER to prevent false positives
        if word_list[-1][:i] in english_set:
            # splits the last elem in the word list at this index and recurses
            check_compound_recursive(word_list[:-1] + [word_list[-1][:i], word_list[-1][i:]], english_set, found_splits)


if __name__ == "__main__":
    evaluate_model()