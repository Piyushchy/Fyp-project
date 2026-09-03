## Comparison

| Model | Params | Disk | Accuracy | Macro-F1 | Macro-P | Macro-R | Latency | vs BERT-base |
|---|---|---|---|---|---|---|---|---|
| TinyBERT (4L-312D) | 14.4M | 55 MB | 92.30% | 88.55% | 86.77% | 91.22% | 2.8 ms | 7.6x smaller |
| DistilBERT (6L-768D) | 67.0M | 255 MB | 93.15% | 89.29% | 87.52% | 91.67% | 10.6 ms | 1.6x smaller |
| MobileBERT (24L-512D) | 24.6M | 94 MB | 93.25% | 89.42% | 87.62% | 91.93% | 16.9 ms | 4.5x smaller |

## Per-emotion F1 (%)

| Model | sadness | joy | love | anger | fear | surprise |
|---|---|---|---|---|---|---|
| TinyBERT | 96.0 | 94.2 | 83.6 | 92.4 | 88.7 | 76.3 |
| DistilBERT | 96.9 | 95.1 | 86.0 | 93.2 | 89.1 | 75.5 |
| MobileBERT | 97.1 | 95.6 | 85.8 | 92.2 | 88.6 | 77.2 |
| _test rows_ | 579 | 688 | 156 | 274 | 224 | 65 |

## Training cost

| Model | Epochs | LR | Train time (CPU) | Throughput |
|---|---|---|---|---|
| TinyBERT | 5 | 5e-05 | 13 min | 722 sentences/s |
| DistilBERT | 3 | 3e-05 | 42 min | 100 sentences/s |
| MobileBERT | 5 | 0.0001 | 47 min | 151 sentences/s |
