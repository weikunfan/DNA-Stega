import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
import os
import math

# 设置环境变量，如果需要的话
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class VAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate, n_z, 
                 encoder_type='dnabert', decoder_type='lstm', load_pretrained=True, mer_value='3',
                 n_heads=8, dim_feedforward=2048):  # 添加transformer相关参数
        super(VAE, self).__init__()
        
        self.encoder_type = encoder_type.lower()
        self.decoder_type = decoder_type.lower()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        
        # 根据编码器类型初始化不同的编码器
        if self.encoder_type == 'dnabert':
            if load_pretrained:
                bert_path = f'/home/fan/Code/VAE_Synthetic_Steganography/pretrained_models/DNAbert_{mer_value}mer'
                print(f'Loading pretrained DNABERT-{mer_value}mer from: {bert_path}')
                self.encoder = transformers.AutoModel.from_pretrained(bert_path, trust_remote_code=True)
            else:
                config = transformers.BertConfig()
                self.encoder = transformers.BertModel(config)
            encoder_hidden_size = self.encoder.config.hidden_size
            
        elif self.encoder_type in ['lstm', 'gru']:
            self.encoder_embedding = nn.Embedding(vocab_size, embed_size)
            rnn_class = nn.LSTM if self.encoder_type == 'lstm' else nn.GRU
            self.encoder = rnn_class(
                input_size=embed_size,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_rate,
                batch_first=True,
                bidirectional=True
            )
            encoder_hidden_size = hidden_dim * 2  # 双向RNN，所以维度翻倍
            
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # 计算潜变量的均值和对数方差的全连接层
        self.hidden_to_mu = nn.Linear(encoder_hidden_size, n_z)
        self.hidden_to_logvar = nn.Linear(encoder_hidden_size, n_z)
        
        # 解码器公共组件
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 根据解码器类型初始化不同的解码器
        if self.decoder_type == 'transformer':
            # Transformer特有的组件
            self.decoder_input_proj = nn.Linear(embed_size + n_z, hidden_dim)
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout_rate)
            
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers
            )
            
            # Transformer的输出层
            self.output_layer = nn.Linear(hidden_dim, vocab_size)
        else:
            # LSTM/GRU解码器
            decoder_rnn_class = nn.LSTM if self.decoder_type == 'lstm' else nn.GRU
            self.decoder = decoder_rnn_class(
                input_size=embed_size + n_z,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_rate,
                batch_first=True
            )
            self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, input_ids, attention_mask=None):
        if self.encoder_type == 'dnabert':
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h_last = encoder_outputs.last_hidden_state[:, 0, :]  # 获取 [CLS] 标记的表示
        else:
            # 对于LSTM/GRU，使用自己的embedding层
            x_embed = self.encoder_embedding(input_ids)
            if self.encoder_type == 'lstm':
                output, (h_n, _) = self.encoder(x_embed)
            else:  # GRU
                output, h_n = self.encoder(x_embed)
            
            # 合并双向RNN的最后一层隐状态
            h_last = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        return h_last
    
    def forward(self, input_ids, attention_mask, x_dec):
        # 编码部分
        h_last = self.encode(input_ids, attention_mask)
        mu = self.hidden_to_mu(h_last)
        logvar = self.hidden_to_logvar(h_last)
        
        # 重参数化
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 准备解码器输入
        batch_size, seq_len = x_dec.size()
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        x_embed_dec = self.embedding(x_dec)
        dec_input = torch.cat([x_embed_dec, z_expanded], dim=2)
        
        # 根据解码器类型进行解码
        if self.decoder_type == 'transformer':
            # Transformer解码
            dec_input = self.decoder_input_proj(dec_input)
            dec_input = self.pos_encoder(dec_input)
            
            # 创建memory和mask
            memory = z.unsqueeze(1)
            zeros = torch.zeros(z.size(0), 1, self.hidden_dim - z.size(1)).to(z.device)
            memory = torch.cat([memory, zeros], dim=2)
            
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x_dec.device)
            
            h_dec = self.decoder(
                dec_input,
                memory=memory,
                tgt_mask=tgt_mask
            )
        else:
            # LSTM/GRU解码
            if self.decoder_type == 'lstm':
                h_dec, _ = self.decoder(dec_input)
            else:  # GRU
                h_dec, _ = self.decoder(dec_input)
        
        # 输出层
        logits = self.output_layer(h_dec)
        log_prob = self.log_softmax(logits)
        
        return log_prob, mu, logvar

    def sample(self, x_dec):
        # 采样函数保持不变，但需要考虑不同的解码器类型
        batch_size, seq_len = x_dec.size()
        n_z = self.hidden_to_mu.out_features
        z = torch.randn(batch_size, n_z).to(x_dec.device)
        
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        x_embed_dec = self.embedding(x_dec)
        dec_input = torch.cat([x_embed_dec, z_expanded], dim=2)
        
        if self.decoder_type == 'transformer':
            dec_input = self.decoder_input_proj(dec_input)
            dec_input = self.pos_encoder(dec_input)
            
            memory = z.unsqueeze(1)
            zeros = torch.zeros(z.size(0), 1, self.hidden_dim - z.size(1)).to(z.device)
            memory = torch.cat([memory, zeros], dim=2)
            
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x_dec.device)
            
            h_dec = self.decoder(
                dec_input,
                memory=memory,
                tgt_mask=tgt_mask
            )
        else:
            if self.decoder_type == 'lstm':
                h_dec, _ = self.decoder(dec_input)
            else:  # GRU
                h_dec, _ = self.decoder(dec_input)
        
        logits = self.output_layer(h_dec)
        log_prob = self.log_softmax(logits)
        
        prob = torch.exp(log_prob)[:, -1, :]
        prob[:, 1] = 0  # 设置特殊符号的概率为0
        prob = prob / prob.sum(dim=1, keepdim=True)
        return torch.multinomial(prob, 1)
