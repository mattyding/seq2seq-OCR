import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from nltk.tokenize import word_tokenize
from settings import DATA_PATH, LATENT_DIM, NUM_SAMPLES, BREAK_CHAR

FREQ_DIRECTORY = "./word-freq-reports/"

DOC_DIRECTORY = "./text-to-predict/"
DOC_TO_DECODE = "sampletext.txt"


""" Preparing Document """
doc_text = []
for line in open(f"{DOC_DIRECTORY}raw-text/{DOC_TO_DECODE}"):
    line = line.replace("\n", " ")
    line = line.replace("\t", "")
    for word in line.split(" "):
        word = word.strip(" ")
        if len(word) > 0:
            doc_text.append(word)

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
num_encoder_tokens = encoder_inputs.shape[2]

translated_doc = []
for word in doc_text:
    print(f'WORD: "{word}"')
    word_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, char in enumerate(word):
        word_data[0, t, input_token_index[char]] = 1.0
    word_data[0, t + 1 :, input_token_index[" "]] = 1.0
    decoded_word = ""
    input_seq = word_data[0:1]
    decoded_word += decode_sequence(input_seq)
    print("DECODED WORD: ", decoded_word)
    translated_doc.append(decoded_word.strip("\n"))

predicted_doc = open(DOC_DIRECTORY + "predicted/P_" + DOC_TO_DECODE, "w+")
predicted_doc.write(" ".join(translated_doc))
predicted_doc.close()
print("Predicted text written to document.")


"""
PyPlot graphs word frequencies
"""
word_counts = {}
for word in translated_doc:
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
plt.bar(range(plot_size), freqs)
plt.xticks(range(plot_size), words, rotation='vertical')
plt.subplots_adjust(bottom=0.15)
plt.savefig(f"{FREQ_DIRECTORY}{DOC_TO_DECODE}_word_freq.png")