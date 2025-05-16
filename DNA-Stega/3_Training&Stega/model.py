import torch
from transformers import AutoTokenizer, AutoModel
import os

def get_model(settings):
    """
    Get the appropriate model based on settings.
    """
    if settings.task == 'text':
        bert_path = '/home/fan/Code/VAE_Synthetic_Steganography/pretrained_models/DNAbert_3mer'
        model = AutoModel.from_pretrained(bert_path, trust_remote_code=True)
        model.to(settings.device)
        return model
    else:
        raise ValueError(f"Unsupported task type: {settings.task}")

def get_tokenizer(settings):
    """
    Get the appropriate tokenizer based on settings.
    """
    if settings.task == 'text':
        bert_path = '/home/fan/Code/VAE_Synthetic_Steganography/pretrained_models/DNAbert_3mer'
        tokenizer = AutoTokenizer.from_pretrained(bert_path, trust_remote_code=True)
        tokenizer.eos_token = tokenizer.sep_token
        tokenizer.eos_token_id = tokenizer.sep_token_id
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.bos_token_id = tokenizer.cls_token_id
        return tokenizer
    else:
        raise ValueError(f"Unsupported task type: {settings.task}")

def get_feature_extractor(settings):
    """
    Get the appropriate feature extractor based on settings.
    """
    if settings.task == 'text':
        return None  # Text task doesn't need a feature extractor
    else:
        raise ValueError(f"Unsupported task type: {settings.task}") 