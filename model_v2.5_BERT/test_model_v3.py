"""
test_model_v3.py
-------------
This program evaluates the trained model and uses it to predict text. This version implements
Google's pre-trained BERT model.

To use, put a folder of text files in the directory specified by DOC_DIRECTORY/RAW_TEXT
see settings.py for more details and to change the location where the program tests.
"""
import os
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from process_coha import clean_text_v2
from settings import DATA_PATH, LATENT_DIM, NUM_SAMPLES, BREAK_CHAR
from settings import FREQ_DIRECTORY, DOC_DIRECTORY, PREDICTED_TEXT, RAW_TEXT, ENGLISH_LEXICON

# for BERT
"""
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import re
from difflib import SequenceMatcher
"""
from enchant.checker import SpellChecker




""" Preppping NLTK and BERT """
spellCheck = SpellChecker("en_UK")  # UK English
ps = PorterStemmer()

""" Preparing English Hashset """
english_words = set()
for line in open(ENGLISH_LEXICON):
    english_words.add(line.strip())

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

model = keras.models.load_model("s2s")

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

for folder in os.listdir(DOC_DIRECTORY + RAW_TEXT):
    print(f"\nCURRENT FOLDER: {folder}")
    #.DS_Store is a file automatically created on Macs
    if folder == ".DS_Store":
        continue
    # makes new folders
    pathlib.Path(DOC_DIRECTORY + PREDICTED_TEXT + folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(DOC_DIRECTORY + "spellcheck/" + folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(FREQ_DIRECTORY + folder).mkdir(parents=True, exist_ok=True)
    for f in os.listdir(DOC_DIRECTORY + RAW_TEXT + folder + "/"):
        if f == ".DS_Store":
            continue
        # This checks if it's already been processed. If you want to re-run, delete the predict file
        if ("P_" + f) in os.listdir(DOC_DIRECTORY + PREDICTED_TEXT + folder + "/"):
            print(f"File Already Exists: {f}")
            continue
        else:
            doc_text = []
            textfile = open(f"./{DOC_DIRECTORY + RAW_TEXT + folder}/{f}")
            try:
                for line in textfile:
                    line = line.replace("\n", " ")
                    line = line.replace("\t", "")
                    line = line.replace("- ", "")
                    for word in line.split(" "):
                        word = clean_text_v2(word)
                        if len(word) > 0:
                            doc_text.append(ps.stem(word))  # word stemming
            except:
                print(f"Unicode Error: {f}")
                continue

            """
            STEP 1: LSTM Model
            """

            num_encoder_tokens = encoder_inputs.shape[2]

            couching = False # couching bool tracks if the next word has already been included

            lstm_doc = []  # original LSTM predicted text
            spellck_doc = [] #spellchecked text
            #bert_copy = []  # copy of text for running Google BERT

            #mask_count = 0  #count for BERT

            for i in range(len(doc_text)):
                if couching:
                    couching = False
                    continue

                word = doc_text[i]

                if word in english_words:
                    lstm_doc.append(word)
                    spellck_doc.append(word)
                    #bert_copy.append(word)

                # couches split-up words
                elif ((i != len(doc_text) - 1) and ((word + doc_text[i+1]) in english_words)):
                    lstm_doc.append(word + doc_text[i+1])
                    spellck_doc.append(word + doc_text[i+1])
                    #bert_copy.append(word + doc_text[i+1])
                    couching = True
                else :
                    try:
                        #print(f'WORD: "{word}"')
                        word_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
                        for t, char in enumerate(word):
                            word_data[0, t, input_token_index[char]] = 1.0
                        word_data[0, t + 1 :, input_token_index[" "]] = 1.0
                        decoded_word = ""
                        input_seq = word_data[0:1]
                        decoded_word += decode_sequence(input_seq)
                        decoded_word = decoded_word.strip("\n")
                        lstm_doc.append(decoded_word)
                        #bert_copy.append("[MASK]")

                        """
                        STEP 2: PyEnchant SpellChecker
                        """
                        suggested = spellCheck.suggest(decoded_word.strip("\n"))
                        spellck_doc.append(suggested[0])
                    except:
                        print(f"Can't read word in {f}: {word}")
                        lstm_doc.append("[...]")
                        spellck_doc.append("[...]")
                        #bert_copy.append("[...]")

                """
                STEP 3: Google BERT
                """
                """
                # Load, train and predict using pre-trained model
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                tokenized_text = tokenizer.tokenize(("".join(bert_copy)))
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                MASKIDS = [i for i, e in enumerate(tokenized_text) if e == "[MASK]"]
                # Create the segments tensors
                segs = [i for i, e in enumerate(tokenized_text) if e == "."]
                segments_ids=[]
                prev=-1
                for k, s in enumerate(segs):
                    segments_ids = segments_ids + [k] * (s-prev)
                    prev=s
                segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
                segments_tensors = torch.tensor([segments_ids])
                # prepare Torch inputs 
                tokens_tensor = torch.tensor([indexed_tokens])
                # Load pre-trained model
                model = BertForMaskedLM.from_pretrained('bert-base-uncased')
                # Predict all tokens
                with torch.no_grad():
                    predictions = model(tokens_tensor, segments_tensors)
                
                pred_words=[]
                for i in range(len(MASKIDS)):
                    preds = torch.topk(predictions[0, MASKIDS[i]], k=50) 
                    indices = preds.indices.tolist()
                    list1 = tokenizer.convert_ids_to_tokens(indices)
                    list2 = suggestedwords[i]
                    simmax=0
                    predicted_token=''
                    for word1 in list1:
                        for word2 in list2:
                            s = SequenceMatcher(None, word1, word2).ratio()
                            if s is not None and s > simmax:
                                simmax = s
                                predicted_token = word1
                    text_original = text_original.replace('[MASK]', predicted_token, 1)
            text_original = predict_word(text_original, predictions, MASKIDS)
            print (text_original)
            """



            predicted_doc = open(DOC_DIRECTORY + PREDICTED_TEXT + folder + "/P_" + f, "w+")
            predicted_doc.write(" ".join(lstm_doc))
            predicted_doc.close()
            spellcheck_doc = open(DOC_DIRECTORY + "spellcheck/" + folder + "/SC_" + f, "w+")
            spellcheck_doc.write(" ".join(spellck_doc))
            spellcheck_doc.close()
            print("File Processed: " + f)

            """
            PyPlot graphs word frequencies
            """
            word_counts = {}
            for word in lstm_doc:
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1

            freqs, words = zip(*sorted(zip(word_counts.values(), word_counts.keys()))) 
            freqs = freqs[::-1] # reverses so largest value comes first
            words = words[::-1]
            plot_size = min(len(word_counts), 30) # caps number of words at 30 max
            if len(word_counts) > 30:
                freqs = freqs[0:30]
                words = words[0:30]
            plt.barh([i for i in range(plot_size)], freqs)
            plt.yticks(range(plot_size), words, rotation='horizontal')
            plt.subplots_adjust(left=0.15)
            plt.savefig(f"{FREQ_DIRECTORY}/{folder}/{f}_word_freq.png")