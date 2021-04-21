import numpy as np
import joblib
import os


def _parse_data(subset, lang="hi"):

    data_path = os.path.join(
        "dakshina_dataset_v1.0/", 
        lang, 
        "lexicons", 
        lang + ".translit.sampled." + subset + ".tsv"
    )

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    input_texts, target_texts = [], []
    input_characters, target_characters = set(), set()

    for line in lines:

        if len(line.split("\t")) < 3:
            continue

        input_text, target_text, _ = line.split("\t")
        target_text = "\t" + target_text + "\n"

        input_texts.append(input_text)
        target_texts.append(target_text)

        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)

        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    if subset == "train":

        if ' ' not in input_characters:
            input_characters.add(' ')
                    
        if ' ' not in target_characters:
            target_characters.add(' ')

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)

        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

        dataset_params = {
            "input_characters": input_characters,
            "target_characters": target_characters,
            "num_encoder_tokens": num_encoder_tokens,
            "num_decoder_tokens": num_decoder_tokens,
            "max_encoder_seq_length": max_encoder_seq_length,
            "max_decoder_seq_length": max_decoder_seq_length,
            "input_token_index": input_token_index,
            "target_token_index": target_token_index,
        }

    else:
        dataset_params = None

    return input_texts, target_texts, dataset_params


def _text_conv(input_texts, target_texts, dataset_params):

    encoder_input_data = np.zeros(
        (
            len(input_texts), 
            dataset_params["max_encoder_seq_length"], 
            dataset_params["num_encoder_tokens"]
        ), 
        dtype="float32"
    )
    decoder_input_data = np.zeros(
        (
            len(input_texts), 
            dataset_params["max_decoder_seq_length"], 
            dataset_params["num_decoder_tokens"]
        ), 
        dtype="float32"
    )
    decoder_target_data = np.zeros(
        (
            len(input_texts), 
            dataset_params["max_decoder_seq_length"], 
            dataset_params["num_decoder_tokens"]
        ), 
        dtype="float32"
    )

    input_token_index = dataset_params["input_token_index"]
    target_token_index = dataset_params["target_token_index"]

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):

        for t, char in enumerate(input_text):
            
            encoder_input_data[i, t, input_token_index[char]] = 1.0

        encoder_input_data[i, t+1:, input_token_index[" "]] = 1.0

        for t, char in enumerate(target_text):
            
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            
            if t > 0:
                decoder_target_data[i, t-1, target_token_index[char]] = 1.0

        decoder_input_data[i, t+1:, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data

def load_lexicons(config):

    save_path = os.path.join(
        "dakshina_dataset_v1.0/", 
        config["data"]["language"], 
        "lexicons", 
        "processed_datasets.pkl"
    )

    if os.path.exists(save_path):
        datasets = joblib.load(save_path)
        train_dataset, val_dataset, test_dataset, dataset_params = datasets
    else:
        train_texts, train_labels, dataset_params = _parse_data("train", config["data"]["language"])
        val_texts, val_labels, _ = _parse_data("dev", config["data"]["language"])
        test_texts, test_labels, _ = _parse_data("test", config["data"]["language"])

        train_dataset = _text_conv(train_texts, train_labels, dataset_params)
        val_dataset = _text_conv(val_texts, val_labels, dataset_params)
        test_dataset = _text_conv(test_texts, test_labels, dataset_params)

        datasets = (train_dataset, val_dataset, test_dataset, dataset_params)
        joblib.dump(datasets, save_path)

    config["data"]["num_encoder_tokens"] = dataset_params["num_encoder_tokens"]
    config["data"]["num_decoder_tokens"] = dataset_params["num_decoder_tokens"]

    return train_dataset, val_dataset, test_dataset
