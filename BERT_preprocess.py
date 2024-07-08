# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_sentences(sentences, tokenizer):
    """
    Tokenizes each sentence in the list of sentences using the provided tokenizer.
    Adds special tokens [CLS] at the start and [SEP] at the end of each sentence.

    Args:
        sentences (list of str): List of sentences to tokenize.
        tokenizer (transformers.BertTokenizer): Pretrained BERT tokenizer.

    Returns:
        list of list of int: List of tokenized and encoded sentences.
    """
    tokenized_texts = [tokenizer.tokenize("[CLS] " + sentence + " [SEP]") for sentence in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    return input_ids

def pad_sequences_custom(input_ids, max_len):
    """
    Pads the sequences to the maximum length specified.
    Sequences longer than max_len are truncated, and shorter sequences are padded with 0s.

    Args:
        input_ids (list of list of int): List of tokenized sentences.
        max_len (int): Maximum length for padding/truncating sequences.

    Returns:
        numpy.ndarray: Array of padded sequences.
    """
    return pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")

def create_attention_masks(input_ids):
    """
    Creates attention masks for each sequence.
    A mask has 1s for tokens and 0s for padding.

    Args:
        input_ids (numpy.ndarray): Array of padded sequences.

    Returns:
        list of list of float: List of attention masks.
    """
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

def create_dataloader(inputs, masks, batch_size):
    """
    Creates a DataLoader for the provided inputs and masks with the specified batch size.

    Args:
        inputs (torch.Tensor): Tensor of input IDs.
        masks (torch.Tensor): Tensor of attention masks.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset.
    """
    data = TensorDataset(inputs, masks)
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader

def prepare_inference_data(sentences, tokenizer, max_len, batch_size=32):
    """
    Preprocesses sentences for inference:
    - Tokenizes and pads sentences
    - Creates attention masks
    - Returns a DataLoader for the processed data

    Args:
        sentences (pandas.DataFrame): DataFrame containing the sentences to preprocess.
        tokenizer (transformers.BertTokenizer): Pretrained BERT tokenizer.
        max_len (int): Maximum length for padding/truncating sequences.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the preprocessed data.
    """
    sentences = sentences['text']
    input_ids = tokenize_sentences(sentences, tokenizer)
    input_ids = pad_sequences_custom(input_ids, max_len)
    attention_masks = create_attention_masks(input_ids)

    inputs = torch.tensor(input_ids)
    masks = torch.tensor(np.array(attention_masks))

    dataloader = create_dataloader(inputs, masks, batch_size)
    
    return dataloader

def prepare_training_data(data, max_len, batch_size, word_embedding, split=True, labels=True):
    """
    Transforms data for classification:
    - Tokenizes and pads sentences
    - Creates attention masks
    - Optionally splits data into train, validation, and test sets
    - Returns DataLoaders for each dataset split, or tensors if not splitting

    Args:
        data (pandas.DataFrame or list of str): DataFrame containing sentences and labels or list of sentences.
        max_len (int): Maximum length for padding/truncating sequences.
        batch_size (int): Batch size for the DataLoader.
        word_embedding (str): Pretrained BERT model name.
        split (bool, optional): Whether to split the data into train, validation, and test sets. Defaults to True.
        labels (bool, optional): Whether the data includes labels. Defaults to True.

    Returns:
        tuple: If split is True, returns DataLoaders for train, test, and validation sets.
        Otherwise, returns tensors for input IDs and attention masks.
    """
    tokenizer = BertTokenizer.from_pretrained(word_embedding, do_lower_case=True)

    if labels:
        text = list(data['Sentences'])
        labels_list = list(data['Label'])     
    else:
        text = data

    input_ids = tokenize_sentences(text, tokenizer)
    input_ids = pad_sequences_custom(input_ids, max_len)
    attention_masks = create_attention_masks(input_ids)

    if split:
        np.random.seed(1)
        sample = np.random.permutation(len(data))
        train_idx = sample[:round(0.7 * len(data))]
        val_idx = sample[round(0.7 * len(data)):round(0.85 * len(data))]
        test_idx = sample[round(0.85 * len(data)):]

        train_inputs = torch.tensor(input_ids[train_idx])
        validation_inputs = torch.tensor(input_ids[val_idx])
        test_inputs = torch.tensor(input_ids[test_idx])

        train_labels = torch.tensor(np.array(labels_list)[train_idx], dtype=torch.long)
        validation_labels = torch.tensor(np.array(labels_list)[val_idx], dtype=torch.long)
        test_labels = torch.tensor(np.array(labels_list)[test_idx], dtype=torch.long)

        train_masks = torch.tensor(np.array(attention_masks)[train_idx])
        validation_masks = torch.tensor(np.array(attention_masks)[val_idx])
        test_masks = torch.tensor(np.array(attention_masks)[test_idx])

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

        return train_dataloader, test_dataloader, validation_dataloader

    else:
        return torch.tensor(input_ids), torch.tensor(attention_masks)