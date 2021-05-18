from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    LSTMCell,
    GRU,
    GRUCell,
    SimpleRNN,
    SimpleRNNCell,
    StackedRNNCells,
    Dense,
)
from tensorflow_addons.seq2seq import (
    BahdanauAttention,
    AttentionWrapper,
    BeamSearchDecoder,
    BasicDecoder,
    tile_batch,
)
from tensorflow_addons.seq2seq.sampler import TrainingSampler
from tensorflow.keras.optimizers import Nadam, Adam, SGD
from tensorflow.keras import Model

from .model_utils import loss_function

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import wandb
import os


class Encoder(Model):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config

        # Layer to convert inputs into embeddings
        self.embedding = Embedding(
            config["data"]["source_vocab_size"], config["model"]["embedding_dim"]
        )

        # Using either LSTM, GRU or Simple RNN as the recurrent layer
        if config["model"]["cell_type"] == "LSTM":
            rec_func = LSTM
        elif config["model"]["cell_type"] == "GRU":
            rec_func = GRU
        else:
            rec_func = SimpleRNN

        # Stacking the recurrent layers
        self.recurrent_layers = {}
        for i in range(config["model"]["encoder"]["layers"]):
            self.recurrent_layers[i] = rec_func(
                config["model"]["encoder"]["units"],
                dropout=config["model"]["encoder"]["dropout"],
                recurrent_dropout=config["model"]["encoder"]["recurrent_dropout"],
                return_sequences=True,
                return_state=True,
            )

    def call(self, x):

        # Passing inputs through embedding layer
        x = self.embedding(x)

        # Passing the embeddings through all recurrent layers
        for i in range(self.config["model"]["encoder"]["layers"]):
            x, *states = self.recurrent_layers[i](x)

        return x, states


class Decoder(Model):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.config = config

        # Layer to convert inputs into embeddings
        self.embedding = Embedding(
            config["data"]["target_vocab_size"], config["model"]["embedding_dim"]
        )

        # Output layer
        self.fc = Dense(config["data"]["target_vocab_size"])

        # Using either LSTMCell, GRU or Simple RNN as the recurrent unit
        if config["model"]["cell_type"] == "LSTM":
            rec_func = LSTMCell
        elif config["model"]["cell_type"] == "GRU":
            rec_func = GRUCell
        else:
            rec_func = SimpleRNNCell

        # Decoder RNN - single cell to sample every time step
        rnn_cells = [
            rec_func(config["model"]["decoder"]["units"])
            for _ in range(config["model"]["decoder"]["layers"])
        ]
        if config["model"]["attention"]:
            rnn_cell = StackedRNNCells(rnn_cells)
            self.attention_mechanism = BahdanauAttention(
                units=config["model"]["decoder"]["units"],
                memory=None,
                memory_sequence_length=config["train"]["batch_size"]
                * [config["data"]["max_length_output"]],
            )
            self.rnn_cell = AttentionWrapper(
                rnn_cell,
                self.attention_mechanism,
                attention_layer_size=config["model"]["decoder"]["units"],
            )
        else:
            self.rnn_cell = StackedRNNCells(rnn_cells)

        # Sampler to sample from softmax output at every time step
        self.sampler = TrainingSampler()

        # Decoder that helps get the output of the network (all time steps)
        self.decoder = BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.fc
        )

    def build_initial_state(self, batch_size, encoder_state, Dtype):
        """
        This function is to be used when attention is used as we need an attention wrapper state.
        """

        # Building the initial state for the decoder based on number of layer and cell type
        if self.config["model"]["decoder"]["layers"] == 1:
            if self.config["model"]["cell_type"] == "LSTM":
                encoder_state = (encoder_state,)
            else:
                encoder_state = tuple(encoder_state)
        else:
            if self.config["model"]["cell_type"] == "LSTM":
                # If there are multiple decoder layers, feed encoder state as initial state to all layers
                states = []
                for i in range(self.config["model"]["decoder"]["layers"]):
                    states += [encoder_state]

                encoder_state = tuple(states)
            else:
                # If there are multiple decoder layers, feed encoder state as initial state to all layers
                states = []
                for i in range(self.config["model"]["decoder"]["layers"]):
                    states += encoder_state

                encoder_state = tuple(states)

        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_size, dtype=Dtype
        )

        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

        return decoder_initial_state

    def get_initial_state(self, encoder_states):
        """
        This function is to be used when attention is not used.
        """

        # Building the initial state for the decoder based on number of layer and cell type
        if self.config["model"]["decoder"]["layers"] == 1:
            states = encoder_states
        else:
            # If there are multiple decoder layers, feed encoder state as initial state to all layers
            if self.config["model"]["cell_type"] == "LSTM":
                states = []
                for i in range(self.config["model"]["decoder"]["layers"]):
                    states += encoder_states
            else:
                states = [
                    encoder_states
                    for i in range(self.config["model"]["decoder"]["layers"])
                ]

        return states

    def call(self, inputs, initial_state):

        # Passing inputs through embedding layer
        x = self.embedding(inputs)

        # Passing the embeddings through the decoder object to get the outputs
        outputs, *states = self.decoder(
            x,
            initial_state=initial_state,
            sequence_length=self.config["train"]["batch_size"]
            * [self.config["data"]["max_length_output"] - 1],
        )
        return outputs


class EncoderDecoder:
    def __init__(self, config):
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Setting the optimizer
        if config["train"]["optimizer"] == "Momentum":
            self.optimizer = SGD(momentum=0.9, nesterov=True)
        elif config["train"]["optimizer"] == "Adam":
            self.optimizer = Adam()
        else:
            self.optimizer = Nadam()

        # Beam Search Object (will be called during inference)
        self.beam_decoder = BeamSearchDecoder(
            self.decoder.rnn_cell,
            beam_width=self.config["model"]["beam_width"],
            output_layer=self.decoder.fc,
            maximum_iterations=config["data"]["max_length_output"],
        )

    @tf.function
    def _train_step(self, inp, targ):

        loss = 0

        # Using gradient tape to track gradients for updates
        with tf.GradientTape() as tape:

            # Passing inputs through the encoder network
            enc_output, states = self.encoder(inp)

            # Removing the end word token
            dec_input = targ[:, :-1]

            # Removing the start word token
            real = targ[:, 1:]

            # Sharing encoder state / outputs with the decoder network
            if self.config["model"]["attention"]:
                self.decoder.attention_mechanism.setup_memory(enc_output)
                decoder_initial_state = self.decoder.build_initial_state(
                    self.config["train"]["batch_size"], states, tf.float32
                )
            else:
                decoder_initial_state = self.decoder.get_initial_state(states)

            # Passing the inputs and the encoder state throgh
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output
            loss = loss_function(real, logits)

        # Accumulating gradients and updating weights
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    @tf.function
    def _validation_step(self, inp, targ):

        # Passing inputs through the encoder network
        enc_output, states = self.encoder(inp)

        # Removing the end word token
        dec_input = targ[:, :-1]

        # Removing the start word token
        real = targ[:, 1:]

        # Sharing encoder state / outputs with the decoder network
        if self.config["model"]["attention"]:
            self.decoder.attention_mechanism.setup_memory(enc_output)
            decoder_initial_state = self.decoder.build_initial_state(
                self.config["train"]["batch_size"], states, tf.float32
            )
        else:
            decoder_initial_state = self.decoder.get_initial_state(states)

        # Passing the inputs and the encoder state throgh
        pred = self.decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = loss_function(real, logits)

        return loss

    def train(self, train_dataset, val_dataset):

        steps_per_epoch = (
            self.config["data"]["num_train_examples"]
            // self.config["train"]["batch_size"]
        )

        for epoch in range(self.config["train"]["epochs"]):

            for batch, ((t_inp, t_targ), (v_inp, v_targ)) in enumerate(
                zip(
                    train_dataset.take(steps_per_epoch),
                    val_dataset[0].take(steps_per_epoch),
                )
            ):

                # Compute and log train and val loss for each batch
                train_batch_loss = self._train_step(t_inp, t_targ)
                val_batch_loss = self._validation_step(v_inp, v_targ)

                wandb.log(
                    {
                        "epoch": epoch,
                        "step": epoch * steps_per_epoch + batch + 1,
                        "train_loss": train_batch_loss.numpy(),
                        "val_loss": val_batch_loss.numpy(),
                    }
                )

            # Compute validation accuracy on the entire validation set every epoch
            val_accuracy = self.evaluate(
                val_dataset[1], "val", write_to_file=False, plot_preds=False
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                }
            )

    def evaluate(self, inputs, title, write_to_file=False, plot_preds=False):

        inference_batch_size = inputs.shape[0]

        # Pass the inputs through the encoder network
        enc_out, states = self.encoder(inputs)

        # Tile the encoder outputs and hidden states for beam search
        if self.config["model"]["attention"]:
            enc_out = tile_batch(enc_out, multiplier=self.config["model"]["beam_width"])
            self.decoder.attention_mechanism.setup_memory(enc_out)
            hidden_state = tile_batch(
                states, multiplier=self.config["model"]["beam_width"]
            )
            hidden_state = self.decoder.build_initial_state(
                self.config["model"]["beam_width"] * inference_batch_size,
                hidden_state,
                tf.float32,
            )
        else:
            if self.config["model"]["cell_type"] != "LSTM":
                states = states[0]

            states = self.decoder.get_initial_state(states)
            hidden_state = tile_batch(
                states, multiplier=self.config["model"]["beam_width"]
            )
            enc_out = tile_batch(enc_out, multiplier=self.config["model"]["beam_width"])

            if self.config["model"]["decoder"]["layers"] == 1:
                hidden_state = (hidden_state,)
            else:
                if self.config["model"]["cell_type"] == "LSTM":
                    hidden_state = tuple(
                        [hidden_state[2 * i], hidden_state[2 * i + 1]]
                        for i in range(int(len(hidden_state) / 2))
                    )
                else:
                    hidden_state = tuple(hidden_state)

        # Start and end tokens for beam search
        start_tokens = tf.fill(
            [inference_batch_size],
            self.config["data"]["target_lang_tokenizer"].word_index["\t"],
        )
        end_token = self.config["data"]["target_lang_tokenizer"].word_index["\n"]

        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        # Running beam search
        outputs, final_state, sequence_lengths = self.beam_decoder(
            decoder_embedding_matrix,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=hidden_state,
        )

        # Getting the output in desired shape
        final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
        beam_scores = tf.transpose(
            outputs.beam_search_decoder_output.scores, perm=(0, 2, 1)
        )

        # Getting numpy arrays from tensors
        final_outputs, beam_scores = final_outputs.numpy(), beam_scores.numpy()

        # Getting the inputs and allowed labels
        labels = self.config["data"][title + "_labels"]
        texts = list(labels.keys())

        # Evaluation loop
        predictions = {
            "Native Script": [],
            "Predicted Latin Script": [],
            "isAccurate": [],
        }
        for i in range(inference_batch_size):
            # Text and allowed labels
            text = texts[i]
            allowed_labels = labels[text]

            # Converting numpy array back to text format
            output = self.config["data"]["target_lang_tokenizer"].sequences_to_texts(
                final_outputs[i]
            )

            # Using the end token to end words; removing spaces
            output = [
                a[: a.index("\n")].replace(" ", "") if "\n" in a else a.replace(" ", "")
                for a in output
            ]

            # Saving input, predicted text and whether the model was accurate
            predictions["Native Script"].append(text)
            predictions["Predicted Latin Script"].append(output[0])
            predictions["isAccurate"].append(output[0] in allowed_labels)

        # Converting to pandas dataframe
        predictions = pd.DataFrame.from_dict(predictions)

        # Computing accuracy
        accuracy = predictions["isAccurate"].mean()

        # Log test accuracy
        if title == "test":
            wandb.log({"test_accuracy": accuracy})

        # Write the pandas dataframe into a file
        if write_to_file:
            path = os.path.join("predictions", wandb.run.name)

            if not os.path.exists(path):
                os.makedirs(path)

            if self.config["model"]["attention"]:
                predictions[["Native Script", "Predicted Latin Script"]].to_csv(
                    path + "/predictions_attention.csv", index=False
                )
            else:
                predictions[["Native Script", "Predicted Latin Script"]].to_csv(
                    path + "/predictions_vanilla.csv", index=False
                )

        return accuracy