from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.data import Dataset

from collections import OrderedDict

import tensorflow as tf
import numpy as np
import joblib
import os


def _parse_data(subset, lang="hi"):

    # Path to read data from
    data_path = os.path.join(
        "dakshina_dataset_v1.0/",
        lang,
        "lexicons",
        lang + ".translit.sampled." + subset + ".tsv",
    )

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    input_texts, target_texts = [], []

    for line in lines:

        # Skip lines that do not meet the expected format
        if len(line.split("\t")) < 3:
            continue

        # Get the input and target word
        input_text, target_text, _ = line.split("\t")

        # Add start and end token to the target word
        target_text = "\t" + target_text + "\n"

        input_texts.append(input_text)
        target_texts.append(target_text)

    return input_texts, target_texts


def _get_valid_labels(texts, labels):
    """
    Since each input word can have multiple possible labels, a dictionary is
    made with the input word as key and a list of all possible labels as value
    """
    valid_labels = OrderedDict()
    for text, label in zip(texts, labels):
        if text not in valid_labels.keys():
            valid_labels[text] = [label[1:-1]]
        else:
            valid_labels[text].append(label[1:-1])

    return valid_labels


def load_lexicons(config):

    # Read each set from the respective file from the dataset
    train_texts, train_labels = _parse_data("train", config["data"]["language"])
    val_texts, val_labels = _parse_data("dev", config["data"]["language"])
    test_texts, test_labels = _parse_data("test", config["data"]["language"])

    # Save all the relevant information that will be used later
    config["data"]["num_train_examples"] = len(train_texts)
    config["data"]["val_texts"] = val_texts
    config["data"]["test_texts"] = test_texts
    config["data"]["val_labels"] = _get_valid_labels(val_texts, val_labels)
    config["data"]["test_labels"] = _get_valid_labels(test_texts, test_labels)

    # Character level tokenization for source language
    source_lang_tokenizer = Tokenizer(char_level=True)
    source_lang_tokenizer.fit_on_texts(train_texts)

    # Character level tokenization for target language
    target_lang_tokenizer = Tokenizer(char_level=True)
    target_lang_tokenizer.fit_on_texts(train_labels)

    # Saving the tokenizers and vocab lengths
    config["data"]["source_vocab_size"] = len(source_lang_tokenizer.word_index) + 1
    config["data"]["target_vocab_size"] = len(target_lang_tokenizer.word_index) + 1
    config["data"]["source_lang_tokenizer"] = source_lang_tokenizer
    config["data"]["target_lang_tokenizer"] = target_lang_tokenizer

    # Tokenizing and padding training data
    train_inputs = source_lang_tokenizer.texts_to_sequences(train_texts)
    train_labels = target_lang_tokenizer.texts_to_sequences(train_labels)
    train_inputs = pad_sequences(train_inputs, padding="post")
    train_labels = pad_sequences(train_labels, padding="post")

    # Saving max sequence lengths
    config["data"]["max_length_input"] = train_inputs.shape[1]
    config["data"]["max_length_output"] = train_labels.shape[1]

    # Tokenizing and padding validation data
    val_inputs = source_lang_tokenizer.texts_to_sequences(val_texts)
    val_labels = target_lang_tokenizer.texts_to_sequences(val_labels)
    val_inputs = pad_sequences(val_inputs, padding="post", maxlen=train_inputs.shape[1])
    val_labels = pad_sequences(val_labels, padding="post", maxlen=train_labels.shape[1])

    # Creating a training dataset for the training loop
    train_dataset = Dataset.from_tensor_slices((train_inputs, train_labels))
    train_dataset = train_dataset.shuffle(len(train_texts)).batch(
        config["train"]["batch_size"], drop_remainder=True
    )

    # Creating a validation dataset for tracking performance in the training loop
    val_dataset = Dataset.from_tensor_slices((val_inputs, val_labels))
    val_dataset = (
        val_dataset.repeat()
        .shuffle(len(val_texts))
        .batch(config["train"]["batch_size"], drop_remainder=True)
    )

    # Creating a tensor with validation data for evaluation in the training loop
    val_X = source_lang_tokenizer.texts_to_sequences(
        list(config["data"]["val_labels"].keys())
    )
    val_X = pad_sequences(val_X, padding="post", maxlen=train_inputs.shape[1])

    # Creating a tensor with test data for final evaluation
    test_X = source_lang_tokenizer.texts_to_sequences(
        list(config["data"]["test_labels"].keys())
    )
    test_X = pad_sequences(test_X, padding="post", maxlen=train_inputs.shape[1])

    return (
        train_dataset,
        (val_dataset, val_X),
        test_X,
    )
