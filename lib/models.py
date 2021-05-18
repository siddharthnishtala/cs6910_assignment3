from tensorflow.keras.layers import Embedding, LSTM, GRU, SimpleRNN
from tensorflow.keras import Model


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
