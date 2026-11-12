
import torch
import torch.nn as nn
import csv
import os

from model import Model
from tokenizer import character_tokenizer
import config


# -------------------------
# 1. Setup
# -------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

torch.manual_seed(1337)


# -------------------------
# 2. Load dataset
# -------------------------

with open("data/tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()


# -------------------------
# 3. Tokenization
# -------------------------

encode,decode,stoi,itos = character_tokenizer(text)

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

# Train / validation split
split_idx = int(config.train_split * len(data))

train_data = data[:split_idx]
val_data = data[split_idx:]

print(f"Vocabulary size: {len(stoi)}")
print(f"Training tokens: {len(train_data)}")
print(f"Validation tokens: {len(val_data)}")


# -------------------------
# 4. Batch generation
# -------------------------

def get_batch(split):

    dataset = train_data if split == 'train' else val_data

    ix = torch.randint(
        len(dataset) - config.block_size - 1,
        (config.batch_size,)
    )

    x = torch.stack([
        dataset[i:i + config.block_size]
        for i in ix
    ])

    y = torch.stack([
        dataset[i + 1:i + config.block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


# -------------------------
# 5. Initialize model
# -------------------------

model = Model(
    vocab_size=len(stoi),
    embedding_size=config.n_embd,
    context_length=config.block_size,
    n_head=config.n_head,
    n_layer=config.n_layer
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# -------------------------
# 6. Estimate loss
# -------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ['train', 'val']:

        losses = torch.zeros(config.eval_iters)

        for k in range(config.eval_iters):

            x, y = get_batch(split)

            logits, loss = model(x, y)

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# -------------------------
# 7. Training loop
# -------------------------

os.makedirs("outputs", exist_ok=True)

training_history = []

for iter in range(config.max_iters):

    # Evaluate periodically
    if iter % config.eval_interval == 0:

        losses = estimate_loss()

        print(
            f"Step {iter}: "
            f"Train Loss {losses['train']:.4f}, "
            f"Val Loss {losses['val']:.4f}"
        )

        training_history.append({
            'step': iter,
            'train_loss': losses['train'],
            'val_loss': losses['val']
        })

    # Sample batch
    xb, yb = get_batch('train')

    # Forward pass
    logits, loss = model(xb, yb)

    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Update weights
    optimizer.step()


# -------------------------
# 8. Save checkpoint
# -------------------------

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab_size': len(stoi),
    'stoi': stoi,
    'itos': itos,
    'config': vars(config)
}

torch.save(checkpoint, "outputs/model.pth")


# -------------------------
# 9. Save training metrics
# -------------------------

with open(
    "outputs/training_loss.csv",
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=['step', 'train_loss', 'val_loss']
    )

    writer.writeheader()
    writer.writerows(training_history)


print("Training completed!")
print("Model saved to outputs/model.pth")
