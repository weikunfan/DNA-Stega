from transformers import AutoTokenizer
import numpy as np
import os
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class Corpus(object):
    def __init__(self, data_path, tokenizer, max_len=200, min_len=5):
        if isinstance(data_path, str):
            data_path = [data_path]
        self._data_path = data_path
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._min_len = min_len
        self.corpus = []
        self.labels = []
        self.sentence_num = 0
        self._build_corpus()

    def _build_corpus(self):
        label = -1
        for data_path in self._data_path:
            label += 1
            with open(data_path, 'r', encoding='utf8') as f:
                sentences = f.readlines()
            for sentence in sentences:
                sentence = sentence.strip()
                tokens = self._tokenizer.tokenize(sentence)
                if (len(tokens) >= self._min_len) and (len(tokens) <= self._max_len):
                    self.corpus.append(sentence)
                    self.labels.append(label)
        self.sentence_num = len(self.corpus)

def split_corpus(data_path, train_path, test_path, max_len=200, min_len=5, ratio=0.8, seed=0, encoding='utf8'):
    with open(data_path, 'r', encoding=encoding) as f:
        sentences = f.readlines()
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [_ for _ in sentences if len(_.split()) <= max_len and len(_.split()) >= min_len]
    np.random.seed(seed)
    np.random.shuffle(sentences)
    train = sentences[:int(len(sentences) * ratio)]
    test = sentences[int(len(sentences) * ratio):]
    with open(train_path, 'w', encoding='utf8') as f:
        for sentence in train:
            f.write(sentence + '\n')
    with open(test_path, 'w', encoding='utf8') as f:
        for sentence in test:
            f.write(sentence + '\n')

class Generator(object):
    def __init__(self, data, tokenizer):
        self._data = np.array(data, dtype=object)
        self._tokenizer = tokenizer

    def build_generator(self, batch_size, shuffle=True):
        indices = list(range(len(self._data)))
        if shuffle:
            np.random.shuffle(indices)
        while True:
            if len(indices) == 0:
                return
            batch_indices = indices[:batch_size]
            indices = indices[batch_size:]
            batch_data = self._data[batch_indices]
            # 使用 tokenizer 对 batch_data 进行编码
            encoded_inputs = self._tokenizer(
                list(batch_data),
                padding='longest',
                truncation=True,
                return_tensors='pt'
            )
            yield encoded_inputs

def loss_function(output, target, mu, logvar, criterion, kl_weight=0.1):
    """
    计算VAE的损失函数
    Args:
        output: 模型输出的重构序列
        target: 目标序列
        mu: 均值向量
        logvar: 对数方差向量
        criterion: 重构损失的criterion (NLLLoss)
        kl_weight: KL散度的权重系数
    Returns:
        total_loss: 总损失
    """
    # 计算重构损失
    recon_loss = criterion(output.view(-1, output.size(-1)), target.reshape(-1))
    
    # 计算KL散度
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / target.size(0)  # 归一化到batch size
    
    # 总损失 = 重构损失 + KL散度 * 权重
    total_loss = recon_loss + kl_weight * kld
    
    return total_loss

class PGD():
    def __init__(self, model, epsilon=1.0, alpha=0.3, steps=3):
        self.model = model
        self.epsilon = epsilon  # 扰动大小的上界
        self.alpha = alpha     # 每步扰动的大小
        self.steps = steps     # 迭代步数
        self.backup = {}
        self.backup_eps = {}
        
    def attack(self, emb_name='embedding', is_first_attack=False):
        """执行PGD对抗攻击"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.backup[name] = param.data.clone()
                    
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)
    
    def project(self, param_name, param_data):
        """将扰动投影到epsilon球内"""
        r = param_data - self.backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.backup[param_name] + r
    
    def restore(self, emb_name='embedding'):
        """恢复模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
