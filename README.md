# Parameter-Efficient Fine-Tuning of Large Language Models

This repository contains the code, results, and report for my M.Tech thesis:
**"Parameter-Efficient Fine-Tuning Techniques on Large Language Models"**,
conducted at IIT Kharagpur under the guidance of Prof. Pawan Goyal and Prof. Arijit De.

## 📄 Overview
The project systematically evaluates multiple Parameter-Efficient Fine-Tuning (PEFT) methods
—including LoRA, QLoRA, AdaLoRA, LoHa, LoKr, IA³, Prefix Tuning, and P-Tuning v2—
on two major NLP tasks:
1. **Summarization** (BillSum dataset) using FLAN-T5-XL and LLaMA-3.2-3B
2. **Classification** (dair-ai/emotion dataset)

## 🚀 Models & Hugging Face Links
| Task            | Model               | PEFT Method | Link |
|-----------------|--------------------|-------------|------|
| Summarization   | FLAN-T5-XL          | AdaLoRA     | [HF Link](https://huggingface.co/...) |
| Summarization   | LLaMA-3.2-3B        | LoKR        | [HF Link](https://huggingface.co/...) |
| Classification  | FLAN-T5-XL          | IA³         | [HF Link](https://huggingface.co/...) |
| Classification  | LLaMA-3.2-3B        | LoRA        | [HF Link](https://huggingface.co/...) |

> All other PEFT variants and configurations are listed in `results/`.

## 📊 Key Results
- **Summarization:** AdaLoRA on T5 achieved ROUGE-1 of **50.54**, training only **0.2478%** of parameters.
- **Classification:** LoRA on LLaMA achieved an F1 score of **0.93**.
- **Efficiency:** QLoRA reduced GPU usage to ~6GB with minimal performance drop.

## 📂 Repository Structure
```
.
├── data/                  # Dataset preprocessing scripts
├── notebooks/             # Jupyter notebooks for experiments
├── src/                   # Training & evaluation scripts
├── results/               # Experimental results & plots
├── report/                # Full thesis PDF & figures
├── scripts/               # Helper scripts
├── requirements.txt       # Python dependencies
└── README.md
```

## 📦 Installation
```bash
git clone https://github.com/<your-username>/parameter-efficient-llm-finetuning.git
cd parameter-efficient-llm-finetuning
pip install -r requirements.txt
```

## 🛠 Usage
### Summarization
```bash
python src/summarization/train_t5.py --config configs/t5_lora.json
```

### Classification
```bash
python src/classification/train_llama.py --config configs/llama_adalora.json
```

## 📑 Citation
If you use this work, please cite:
```
Beerukuri, Abhinay. "Parameter-Efficient Fine-Tuning Techniques on Large Language Models." M.Tech Thesis, IIT Kharagpur, 2025.
```

## 📜 License
This repository is licensed under the MIT License.
