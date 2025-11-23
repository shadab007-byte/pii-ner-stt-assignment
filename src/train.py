import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_length", type=int, default=64)  # REDUCED for speed
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def evaluate(model, dev_dl, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dev_dl:
            input_ids = torch.tensor(batch["input_ids"], device=device)
            attention_mask = torch.tensor(batch["attention_mask"], device=device)
            labels = torch.tensor(batch["labels"], device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    model.train()
    return total_loss / max(1, len(dev_dl))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {args.device}, Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    dev_ds = PIIDataset(args.dev, tokenizer, LABELS, max_length=args.max_length, is_train=True)
    
    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id))
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id))

    model = create_model(args.model_name)
    model.to(args.device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    print(f"Total steps: {total_steps}\n")
    best_dev_loss = float('inf')

    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_train = running_loss / max(1, len(train_dl))
        dev_loss = evaluate(model, dev_dl, args.device)
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Dev={dev_loss:.4f}")
        
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            print(f"  → Best! Saving...")
            model.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)

    print(f"\nDone! Best dev loss: {best_dev_loss:.4f}")


if __name__ == "__main__":
    main()