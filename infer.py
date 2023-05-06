import re
import argparse

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from text import phoneme_to_ids

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def text_to_phonemes(text, lexicon):
    text = text.lower()
    words = re.split(r"([,;.\-\?\!\s+])", text)
    
    phonemes = []
    for word in words:
        if word in lexicon:
            phoneme = lexicon[word]
            phonemes += phoneme
        elif len(word.strip()) == 0:
            continue
        else:
            pass
            
    phoneme_ids = phoneme_to_ids(" ".join(phonemes))
    phoneme_ids = torch.tensor(phoneme_ids).reshape(1, len(phoneme_ids))
    
    text_lens = torch.tensor([len(phoneme_ids[0])])
    batch = [("test", ), text, phoneme_ids, text_lens, max(text_lens)]
    
    return batch
    
    
if __name__ == "__main__":
    preprocess_config = "config/preprocess.yaml"
    model_config = "config/model.yaml"
    train_config = "config/train.yaml"
    
    preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    preprocess_config["path"]["preprocessed_path"] = "processed_data"
    
    class args:
        restore_step = 76000
    _args = args()
    
    text = "hôm nay là ngày mùng sáu tháng năm năm hai nghìn hai mươi ba"
    
    model = get_model(_args, configs, device="cpu", train=False)
    vocoder = get_vocoder(model_config, device="cpu")
    
    lexicon = read_lexicon("data/lexicon_ipa")
    
    input_batch = text_to_phonemes(text, lexicon)
    output = model(
        *(input_batch[2:]), device="cpu"
    )
    
    synth_samples(
        input_batch,
        output,
        vocoder,
        model_config,
        preprocess_config,
        train_config["path"]["result_path"],
    )