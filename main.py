from lib.utils import load_lexicons

CONFIG = {
    "data": {
        "language": "hi"
    },
    "model": {
        "cell_type": "LSTM",
        "encoder_lstm_dim": 32,
        "decoder_lstm_dim": 32
    },
    "train": {
        "optimizer": "Adam",
        "epochs": 5,
        "batch_size": 32
    }
}

train_dataset, val_dataset, test_dataset = load_lexicons(CONFIG)
