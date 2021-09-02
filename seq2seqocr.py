import os
import re
import json
import pickle
import string
import numpy as np
from tensorflow import keras
from threading import Thread, Lock

from general_util import clean_text
from settings import *

CLASSIFICATION_TOKEN = "[CLS]"
UNIDENTIFIABLE_TOKEN = "[...]"

def main():
    model = Seq2SeqOCR('s2s')
    print(model.predict("loods"))

    #print(model.predict('clty'))
    #print(model.predict('affiftant'))
    return

    with open("SAMPLE_FILE.txt", 'r') as f:
        sample_text = f.read()
        cleaned_text = model.preprocess(sample_text)
        print(cleaned_text)

        #ew_text = []
        #for word in cleaned_text.split():
            #if word.startswith(CLASSIFICATION_TOKEN):
                #predicted_word = model.predict(word[len(CLASSIFICATION_TOKEN):])
                #print(word[len(CLASSIFICATION_TOKEN):], predicted_word)
                #new_text.append(predicted_word)
            #else:
                #new_text.append(word)
        #print("".join(new_text))


class Seq2SeqOCR:
    def __init__(self, model_path : str):
        super(Seq2SeqOCR, self).__init__()

        self.input_texts, target_texts, input_chars, target_chars = parse_training_data()
        self.Input = CharacterTable(self.input_texts, input_chars)
        self.Target = CharacterTable(target_texts, target_chars)

        self.english_lexicon, self.common_lexicon = retrieve_lexicons()

        self.memoized_words = {}  # for preprocessing memoization

        model = keras.models.load_model(model_path)

        encoder_inputs = model.input[0]  # input 1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm 1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = keras.Model(encoder_inputs, encoder_states)

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
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

    def predict(self, input_text : str) -> str:
        encoded_word = self.encode_sequence(input_text)
        decoded_word = self.decode_sequence(encoded_word)
        return decoded_word

    def encode_sequence(self, input_text : str) -> np.array:
        word_encoding = np.zeros((1, self.Input.max_seq_length, self.Input.num_tokens), dtype="float32")
        for t, char in enumerate(input_text):
            word_encoding[0, t, self.Input.token_index[char]] = 1.0
        word_encoding[0, t + 1 :, self.Input.token_index[" "]] = 1.0
        return word_encoding
    
    def decode_sequence(self, input_seq : np.array) -> str:
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.Target.num_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.Target.token_index[BREAK_CHAR]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.Target.reverse_token_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == ENDSEQ_CHAR or len(decoded_sentence) > self.Target.max_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.Target.num_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence

    
    def preprocess(self, input_text : str) -> str:
        """
        Preprocesses the input text by removing punctuation and correcting spacing errors.
        Saves the processed text to an intermediate file and also returns processed texts.
        """

        # Clean the input text
        input_text = clean_text(input_text.lower())
        # Split the input text into words
        split_text = input_text.split()

        preprocessed_text = []
        couching = False # couching bool tracks if the next word has already been included

        for i in range(len(split_text)):
            if couching:  # next word has been included; skips it
                couching = False
                continue

            word = split_text[i]

            # 1. Recognizable Word: if word in english lexicon set, does not alter it
            if word in self.english_lexicon:
                preprocessed_text.append(word)
            # 2. Forward Couching: if the next two entries make a word, couches them
            elif ((i != len(split_text) - 1) and ((word + split_text[i+1]) in self.english_lexicon)):
                preprocessed_text.append(word + split_text[i+1])
                couching = True
            # 3. Backward Couching: if the previous entry and the current word make a word, couches them
            elif (i != 0) and ((split_text[i-1] + word) in self.english_lexicon):
                preprocessed_text.append(split_text[i-1] + word)
            # 4. Missing Spaces: if the current word can be split into valid words(with avg length > 3)
            elif len(check_compound(word, self.english_lexicon, self.memoized_words)) != len(word):
                # if cannot find valid split, returns original word (same length)
                preprocessed_text.append(check_compound(word, self.common_lexicon, self.memoized_words))
            # 5. Too Long (likely compound): If larger than max_seq_length, returns original word
            elif (len(word) > MAX_SEQ_LENGTH):
                preprocessed_text.append(UNIDENTIFIABLE_TOKEN)
            # 6. Unknown and Inferable: if other pre-checks don't pass, feeds it to the seq2seq model
            else:
                preprocessed_text.append(CLASSIFICATION_TOKEN + word)
                #preprocessed_text.append(CLASSIFICATION_TOKEN)
        
        return " ".join(preprocessed_text)


                


"""
Model Preparation Scripts
"""
class CharacterTable:
    def __init__(self, text_set, char_set):
        super(CharacterTable, self).__init__()
        self.characters = sorted(list(char_set))
        self.num_tokens = len(char_set)
        self.max_seq_length = MAX_SEQ_LENGTH #max([len(txt) for txt in text_set])
        self.token_index = dict([(char, i) for i, char in enumerate(char_set)])
        self.reverse_token_index = dict((i, char) for char, i in self.token_index.items())

def parse_training_data():
    input_characters = set()
    target_characters = set()
    for c in string.ascii_lowercase:
        input_characters.add(c)
        target_characters.add(c)
    target_characters.add(BREAK_CHAR)
    target_characters.add(ENDSEQ_CHAR)
    input_characters.add(' ')
    target_characters.add(' ')

    input_texts = []
    target_texts = []
    with open(DATA_PATH, 'r', encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]  # last line has no BREAK_CHAR
        for line in lines:
            # tab is our break char and newline is end seq char
            input_text, target_text = line.split(BREAK_CHAR)
            target_text = BREAK_CHAR + target_text + ENDSEQ_CHAR
            input_texts.append(input_text)
            target_texts.append(target_text)
    
    return input_texts, target_texts, input_characters, target_characters


def retrieve_lexicons():
    with open(ENGLISH_LEXICON_PKL, "rb") as f1, open(COMMON_ENG_LEXICON_PKL, "rb") as f2:
        english_lexicon = pickle.load(f1)
        common_lexicon = pickle.load(f2)
        f1.close()
        f2.close()
    return english_lexicon, common_lexicon

"""
Preprocessing Scripts
"""
def memoize_check_compound(f):
    def inner(word, common_lexicon, memory):
        if word not in memory:
            memory[word] = f(word, common_lexicon, memory)
        return memory[word]
    return inner

@memoize_check_compound
def check_compound(word : str, english_lexicon : set, memoized_words : dict):
    """
    Checks if a word is a compound word and returns the split word if it is. Does so by 
    using the recursive function check_compound_recursive
    """
    # list containing all of the recursively found splittings
    found_splits = []

    check_compound_recursive([word], english_lexicon, found_splits)

    # no splits found
    if len(found_splits) == 0:
        return word
    
    # returns the splitting with fewest total splits (largest word-sections)
    fewest_splits = min(found_splits, key=len)

    # check to make sure it didn't find 2-word splits throughout (rejects < 4 letter average)
    if len(word) / len(fewest_splits) < 4:
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

    def populate_compound_memory(memoized_words : dict) -> None:
        """
        Populates the memoization set with several common short-word combinations.
        """
        memoized_words['inthe'] = 'in the'
        memoized_words['ofthe'] = 'of the'
        memoized_words['forthe'] = 'for the'
        memoized_words['canbe'] = 'can be'
        memoized_words['goto'] = 'go to'






if __name__ == '__main__':
    main()