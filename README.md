# PII Entity Recognition for STT Transcripts

Token-level NER model for detecting PII entities in noisy Speech-to-Text transcripts.

## 🎯 Results

### Performance Metrics
```
Per-Entity F1 Scores:
- CREDIT_CARD:  1.000
- PHONE:        0.964
- EMAIL:        1.000
- PERSON_NAME:  1.000
- DATE:         1.000
- CITY:         1.000
- LOCATION:     1.000

Macro F1: 0.995
PII Precision: 0.991 ✅ (Target: ≥0.80)
PII Recall: 0.991
PII F1: 0.991
```

### Latency
```
p50: 25.02 ms
p95: 38.54 ms
(Target: ≤20ms - can be optimized with ONNX Runtime)
```

## 🏗️ Architecture

- **Model:** DistilBERT-base-uncased (66M parameters)
- **Task:** Token classification with BIO tagging
- **Max Length:** 64 tokens
- **Training:** 5 epochs, batch_size=16, lr=3e-5

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Generate Synthetic Data
```bash
python generate_data.py
# Generates 600 train + 120 dev + 30 test examples
```

### Train Model
```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out \
  --max_length 64
```

### Generate Predictions
```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json \
  --max_length 64
```

### Evaluate
```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

### Measure Latency
```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50 \
  --max_length 64
```

## 📁 Project Structure
```
├── src/
│   ├── dataset.py          # Data loading & BIO tagging
│   ├── labels.py           # Entity labels and PII mapping
│   ├── model.py            # Model initialization
│   ├── train.py            # Training script
│   ├── predict.py          # Inference script
│   ├── eval_span_f1.py     # Evaluation metrics
│   └── measure_latency.py  # Latency measurement
├── data/
│   ├── train.jsonl         # Training data (600 examples)
│   ├── dev.jsonl           # Dev data (120 examples)
│   └── test.jsonl          # Test data (30 examples)
├── out/
│   ├── dev_pred.json       # Dev set predictions
│   └── test_pred.json      # Test set predictions
├── generate_data.py        # Synthetic data generator
├── requirements.txt        # Dependencies
└── README.md
```

## 📊 Data Format

### Input (JSONL)
```json
{
  "id": "utt_0001",
  "text": "my credit card is 4242 4242 4242 4242",
  "entities": [
    {"start": 18, "end": 37, "label": "CREDIT_CARD"}
  ]
}
```

### Output (JSON)
```json
{
  "utt_0001": [
    {"start": 18, "end": 37, "label": "CREDIT_CARD", "pii": true}
  ]
}
```

## 🎯 Key Features

1. **High Accuracy:** Near-perfect F1 score (0.995)
2. **PII Safety:** High precision (0.991) for sensitive data
3. **Synthetic Data:** Realistic STT patterns with noise
4. **BIO Tagging:** Robust sequence labeling
5. **Span Decoding:** Improved merging and filtering

## 🔧 Design Decisions

- **DistilBERT:** Balance of speed and accuracy
- **Max Length 64:** Optimized for most utterances
- **Conservative Filtering:** High precision for PII safety
- **Gradient Clipping:** Stable training
- **Learning Rate Warmup:** Better convergence

## ✅ Targets Achieved

- ✅ **PII Precision ≥ 0.80:** Achieved 0.991
- ⚠️ **p95 Latency ≤ 20ms:** 38.54ms (can be optimized with ONNX)

## 🚀 Optimization Notes

To achieve <20ms latency:
- Use ONNX Runtime (2-3x speedup)
- Reduce max_length to 48
- Use quantization

## 👤 Author

Shadab - ML Assignment Submission
```
