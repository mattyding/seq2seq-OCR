"""
File: train_model.py
--------------------
This file builds the seq2seq architecture and trains the model.
"""
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from settings import DATA_PATH, BREAK_CHAR, ENDSEQ_CHAR, MAX_SEQ_LENGTH
from settings import BATCH_SIZE, EPOCHS, LATENT_DIM, NUM_SAMPLES
from settings import SAVED_MODEL
from general_util import clean_text_no_spaces


DATA= 'training-sets/forced_errors_1.txt'


def train_model(training_data, saved_model=None):
    """
    Data Preparation
    """

    input_characters = set()
    target_characters = set()
    for c in string.ascii_lowercase:
        input_characters.add(c)
        target_characters.add(c)
    # tab ('\t') is our break char and newline ('\n') is end seq char
    target_characters.add(BREAK_CHAR)
    target_characters.add(ENDSEQ_CHAR)

    input_texts = []
    target_texts = []
    with open(training_data, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    for line in lines[: min(NUM_SAMPLES, len(lines) - 1)]:
        
        input_text, target_text = line.split(BREAK_CHAR)
        target_text = BREAK_CHAR + target_text + ENDSEQ_CHAR
        input_texts.append(input_text)
        target_texts.append(target_text)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = MAX_SEQ_LENGTH
    max_decoder_seq_length = MAX_SEQ_LENGTH

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)


    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

    """
    Building Model
    """

    # Define an input sequence and process it.
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    encoder = keras.layers.LSTM(LATENT_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    """
    Training
    """
    if saved_model:
        model = keras.models.load_model(saved_model)

    earlystop = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        validation_split=0.2,
        callbacks=[earlystop, earlystop]
    )
    # save trained model weights
    model.save(SAVED_MODEL)
    #model.save_weights(SAVED_MODEL)


if __name__ == '__main__':
    train_model(DATA, saved_model=SAVED_MODEL)