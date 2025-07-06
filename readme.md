# 🧠 Language Does Not Equal Cognition: Uniform Neural Patterns in Role-Conditioned Medical LLMs

## 📘 Overview

This repository supports the paper:

**"Language Does Not Equal Cognition: Uniform Neural Patterns in Role-Conditioned Medical LLMs"**  

## 📘 Abstract
The decision-making behavior of clinical doctors exhibits systematic differences due to their professional qualifications, educational background, and clinical experience. This hierarchical cognitive structure is a core component of the modern medical system. With the rapid application of Large Language Models (LLMs) in medical contexts, "Prompt-based Role Playing" has become a key technological approach for simulating expert thinking and enhancing the credibility of generated outputs. However, can LLMs truly mimic clinical doctors effectively through role-playing? In this study, we systematically evaluate the effectiveness of Prompt-based Role Playing in multiple medical reasoning tasks, encompassing accuracy assessments, neural activation analysis, hidden layer similarity analysis, and cross-role neural masking experiments. The results indicate that the influence of different role prompts on model behavior primarily manifests in language style modulation, rather than triggering changes in deep reasoning pathways or structural cognitive modeling. Our experiments reveal that current LLMs still demonstrate a high degree of cognitive homogeneity in understanding medical roles, lacking the ability to simulate the cognitive differentiation of clinical experts. This finding challenges the implicit assumption of "language-equivalent cognition" in current medical AI systems, which treat LLMs as medical agent simulations for medical knowledge reasoning. This conclusion poses new challenges for the development of medical AI, emphasizing the need to shift from imitating language style to modeling doctors' cognitive processes, aiming to build intelligent LLM-based medical systems with professional-level cognitive capabilities.

---

## 🗂️ Directory Structure

```bash
.
├── dataset/                  # (You provide) test sets from MedQA, MedMCQA, MMLU-Med
├── NeuronAnalysis/          # Core experimental pipeline
│   ├── data_loader.py
│   ├── model_utils.py
│   ├── prompt_effects.py
│   ├── utils.py
│   ├── exp1_P.py / exp1_sig.py          # Exp1: QA Accuracy comparison
│   ├── exp2_P.py / exp2_sig.py          # Exp2: JSD divergence analysis
│   ├── exp3_P.py / exp3_sig.py          # Exp3: CKA/PCA for hierarchy perception
│   ├── exp4.1_P.py / exp4.1_sig.py      # Exp4.1: Role-specific neuron masking
│   ├── exp5cross_P.py / exp5cross_sig.py
│   ├── exp5cross.py                     # Shared logic for Exp5
├── Bloomclarify.py         # Bloom-level QA classification module
├── Myutils.py              # I/O, formatting, and shared utilities
├── requirements.txt        # Python environment requirements
└── README.md               # This file

```
## 🚀 Getting Started

### 🟦 1 Installation

Ensure Python 3.10+ is available. Then install dependencies:

```bash
pip install -r requirements.txt
```
### 🟦 2 Model Setup

This project uses instruction-tuned LLMs (mainly Qwen2.5 series). You may need:

Local weights for Qwen2.5-7B/14B/32B/72B-Instruct

Optionally: API access to GPT-4o, Deepseek-R1

### 🟦 3 Run Example

Each experiment has a main logic file and a significance test file. For example:

```bash
cd NeuronAnalysis

python exp1_P.py       # Run accuracy evaluation
```