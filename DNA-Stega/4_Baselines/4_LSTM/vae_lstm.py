import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lm import KL_gradient_rep, m_rep  # 从原始lm.py导入需要的函数

class VAE_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, latent_dim, num_layers, dropout_rate):
        super(VAE_LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            embed_size, 
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # VAE latent space
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            embed_size + latent_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        
    def encode(self, x):
        # Embed input
        embedded = self.embedding(x)
        
        # Encode
        _, (hidden, _) = self.encoder_lstm(embedded)
        
        # Get final hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        
        return mu, log_var
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, x, z):
        # Embed input
        embedded = self.embedding(x)
        
        # Expand z to match sequence length
        z_expanded = z.unsqueeze(1).expand(-1, embedded.size(1), -1)
        
        # Concatenate embedding and latent vector
        decoder_input = torch.cat([embedded, z_expanded], dim=2)
        
        # Decode
        output, _ = self.decoder_lstm(decoder_input)
        
        # Project to vocabulary
        logits = self.output_layer(output)
        
        return logits
    
    def forward(self, x, logits=False):
        # Convert input to long tensor
        x = x.long()
        
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample latent vector
        z = self.reparameterize(mu, log_var)
        
        # Decode
        logits_out = self.decode(x, z)
        
        # Return based on logits flag
        if isinstance(logits, bool) and logits:
            return logits_out, mu, log_var
        else:
            return self.log_softmax(logits_out), mu, log_var
            
    def sample_beg(self, x, temperature=1.0):
        with torch.no_grad():
            # Get latent vector
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            
            # Get probabilities
            log_prob = self.decode(x, z)
            prob = torch.exp(self.log_softmax(log_prob/temperature))[:, -1, :]
            
            # Zero out special tokens
            prob[:, 0:3] = 0
            
            # Normalize
            prob = prob / prob.sum()
            
            return torch.multinomial(prob, 1)
            
    def sample_rep(self, x, Tol, vocabulary, rep):
        with torch.no_grad():
            # Get latent vector
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            
            # Get probabilities
            log_prob = self.decode(x, z)
            prob = torch.exp(self.log_softmax(log_prob))[:, -1, :]
            
            prob[:, 1] = 0
            prob = prob / prob.sum()
            
            # Get KL gradient
            K_gra_16 = KL_gradient_rep(x, Tol, vocabulary, rep)
            sum_k = sum(sum(np.abs(K_gra_16)))
            K_gra_16_ = K_gra_16 / sum_k
            
            pp = m_rep(prob, K_gra_16_, rep)
            
            return torch.multinomial(pp, 1) 