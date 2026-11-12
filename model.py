import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):

    def __init__(self,head_size,embedding_size,context_length):

        super().__init__()

        self.query=nn.Linear(embedding_size,head_size,bias=False)
        self.key=nn.Linear(embedding_size,head_size,bias=False)
        self.value=nn.Linear(embedding_size,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(0.0)

    def forward(self,x):
       
        k=self.key(x)  
        q=self.query(x)
        B,T,H=k.shape
        attention_block=q@k.transpose(-2,-1)*H**-0.5# (B, T, C) @ (B, C, T) -> (B, T, T)
                
        attention_block = attention_block.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        attention_block = F.softmax(attention_block, dim=-1) # (B, T, T)
        attention_block = self.dropout(attention_block)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = attention_block @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, head_size, embedding_size, context_length):
        super().__init__()
        n_heads=embedding_size//head_size
        self.heads=nn.ModuleList([Attention(head_size,embedding_size, context_length) for _ in range(n_heads)])
        self.projection=nn.Linear(embedding_size,embedding_size)
        self.dropout=nn.Dropout(0.0)

    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.projection(out)
        out=self.dropout(out)
        return out


class MLP(nn.Module):

    def __init__(self,embedding_size):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(embedding_size,4*embedding_size),
            nn.ReLU(),
            nn.Linear(4*embedding_size,embedding_size),
            nn.Dropout(0.0)
        )
    def forward(self,x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self,embedding_size,no_head, context_length):
        super().__init__()
        self.sa=MultiHeadAttention(embedding_size//no_head,embedding_size, context_length)
        self.ffwd=MLP(embedding_size)
        self.ln1=nn.LayerNorm(embedding_size)
        self.ln2=nn.LayerNorm(embedding_size)

    def forward(self,x):
          
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)
        return x


    
class Model(nn.Module):

    def __init__(self, vocab_size, embedding_size,context_length,n_head,n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding=nn.Embedding(context_length,embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)
        self.ln_f = nn.LayerNorm(embedding_size) # final layer norm
        self.blocks = nn.ModuleList([
           Block(embedding_size, n_head, context_length)
            for i in range(n_layer)
              ])
        
    

    def forward(self,x,y=None):
        b,t=x.shape
        pos_emb = self.positional_embedding(
          torch.arange(t, device=x.device)
        )
        
        x=self.embedding(x)+pos_emb
        for block in self.blocks:
            x=block(x)

        x=self.ln_f(x)


        logits=self.lm_head(x)


        if y is None:
            loss=None
        else:
            b,t,c=logits.shape
            loss=F.cross_entropy(logits.view(b*t,c),y.view(b*t))
        return logits,loss


