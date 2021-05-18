from lib.utils import load_lexicons
import wandb


hyperparameter_defaults = dict(
    cell_type="LSTM",
    embedding_dim=64,
    attention=True,
    encoder_layers=1,
    encoder_dropout=0.0,
    encoder_rec_dropout=0.0,
    decoder_layers=1,
    hidden_units=64,
    beam_width=3,
    optimizer="Nadam",
    epochs=20,
    batch_size=64,
)

wandb.init(
    config=hyperparameter_defaults, entity="iitm-cs6910", project="CS6910-Assignment-3"
)
config = wandb.config

# Set the run name
name = config["cell_type"] + "_"
name += "emb(" + str(config["embedding_dim"]) + ")_"
name += "enc(" + str(config["encoder_layers"])
name += ", " + str(config["hidden_units"])
name += ", " + str(config["encoder_dropout"])
name += ", " + str(config["encoder_rec_dropout"]) + ")_"
name += "dec(" + str(config["decoder_layers"])
name += ", " + str(config["hidden_units"]) + ")_"
name += "bw(" + str(config["beam_width"]) + ")_"
name += config["optimizer"] + "_"
name += "ep(" + str(config["epochs"]) + ")_"
name += "bs(" + str(config["batch_size"]) + ")_"
name += "att_" if config["attention"] else ""

wandb.run.name = name[:-1]

CONFIG = {
    "data": {
        "language": "hi",
    },
    "model": {
        "cell_type": config["optimizer"],
        "embedding_dim": config["embedding_dim"],
        "attention": config["attention"],
        "encoder": {
            "layers": config["encoder_layers"],
            "units": config["hidden_units"],
            "dropout": config["encoder_dropout"],
            "recurrent_dropout": config["encoder_rec_dropout"],
        },
        "decoder": {
            "layers": config["decoder_layers"],
            "units": config["hidden_units"],
        },
        "beam_width": config["beam_width"],
    },
    "train": {
        "optimizer": config["optimizer"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
    },
}

# Loading the datasets
train_dataset, val_dataset, test_dataset = load_lexicons(CONFIG)
