def character_tokenizer(text):
    char_list=list(sorted(set(text)))
    stoi={ch:i for i,ch in enumerate(char_list)}
    itos={i:ch for i,ch in enumerate(char_list)}
    encode=lambda s:[stoi[c] for c in s]
    decode=lambda l:"".join([itos[i] for i in l])

    return encode,decode,stoi,itos
