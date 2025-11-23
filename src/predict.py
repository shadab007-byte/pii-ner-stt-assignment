import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            if current_label:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue
        
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        if "-" not in label:
            continue
            
        prefix, ent_type = label.split("-", 1)
        
        if prefix == "B":
            if current_label:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label:
        spans.append((current_start, current_end, current_label))

    # Merge adjacent same-type spans
    merged = []
    for s, e, lab in spans:
        if merged and merged[-1][2] == lab and s - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], e, lab)
        else:
            merged.append((s, e, lab))
    
    # Filter very short spans (except DATE)
    filtered = [(s, e, lab) for s, e, lab in merged if lab == "DATE" or e - s >= 3]
    
    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir if not args.model_name else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(text, return_offsets_mapping=True, truncation=True, 
                          max_length=args.max_length, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()

            with torch.no_grad():
                out = model(input_ids=enc["input_ids"].to(args.device), 
                           attention_mask=enc["attention_mask"].to(args.device))
                pred_ids = out.logits[0].argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            ents = [{"start": int(s), "end": int(e), "label": lab, "pii": bool(label_is_pii(lab))} 
                   for s, e, lab in spans]
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    total_ents = sum(len(e) for e in results.values())
    pii_ents = sum(sum(1 for x in e if x["pii"]) for e in results.values())
    print(f"\nPredictions: {len(results)} utterances, {total_ents} entities ({pii_ents} PII)")


if __name__ == "__main__":
    main()