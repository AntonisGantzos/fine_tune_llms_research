# Legal LLM Fine-Tuning: Data Strategy

## Project Overview

**Objective:** Fine-tune General Purpose LLMs (Llama 3, Mistral) on consumer hardware using PEFT/QLoRA for legal contract review.

### Specific Tasks
- **Task 1:** Risk Clause Recognition (Binary Classification/Identification).
- **Task 2:** Structured Entity Extraction (JSON Generation).

---

## 1. Dataset File Roles

The **Contract Understanding Atticus Dataset (CUAD) v1** is distributed in multiple formats. For the specific goals of Generative LLM fine-tuning, the files play the following roles:

### A. Training Data Source
* **File:** `master_clauses.csv` (Approx. 80-90% of rows)
* **Role:** Primary Instruction Tuning Corpus.
* **Justification:**
    * **Aligned Pairs:** Contains direct mapping between "Text Context" (Input) and "Human-Input Answer" (Target Output) required for Supervised Fine-Tuning (SFT).
    * **Normalization:** Answers have been normalized by experts (e.g., transforming "8th day of May" → "5/8/2014"), allowing effective learning of Task 2.
    * **Efficiency:** Aggregates all 510 contracts and 41 categories into a single tabular format, simpler to convert to JSONL than parsing scattered text files.

### B. Validation & Testing Source
* **File:** `master_clauses.csv` (Approx. 10-20% of rows, held-out)
* **Role:** Unseen Evaluation Set.
* **Justification:**
    * **No Official Split:** CUAD v1 does not provide a pre-defined test set.
    * **Leakage Prevention:** Validation must occur at the **Contract Level**, ensuring the model never sees the same contract in both training and testing.
    * **Metric Consistency:** Using the same schema allows for continuous calculation of Exact Match (EM) and F1 scores.

### C. Ground Truth
* **File:** `master_clauses.csv` (Specific Columns: `[Category] Answer`)
* **Role:** The "Gold Label" for loss calculation and evaluation.
* **Justification:**
    * **Expert Annotation:** Labels underwent rigorous multi-step review (Law Student → Keyword Search → Attorney Review).
    * **Task 1 Truth:** Binary presence of the clause or text span in the "Answer" column.
    * **Task 2 Truth:** Normalized string (e.g., "Nevada", "Yes", "12/31/2021").

---

## 2. Omitted / Irrelevant Files

The following components of the CUAD v1 download should be excluded to optimize for consumer hardware and text-generation objectives.

| File / Folder | Action | Reason for Omission |
| :--- | :--- | :--- |
| `full_contracts_pdf/` | **Omit** | **Multimodal Overhead:** Llama 3/Mistral are text-based. Using PDFs requires OCR, introducing noise. The CSV already contains high-quality extraction. |
| `full_contracts_txt/` | **Omit** | **Redundancy:** Relevant clauses are already extracted. Feeding full raw contracts (>50k tokens) exceeds standard consumer context windows. |
| `Label Report - *.xlsx` | **Omit** | **Redundancy:** These 28 Excel files are subsets of the Master CSV. Processing them individually is inefficient. |
| `CUAD_v1.json` | **Omit** | **Wrong Architecture:** Follows SQuAD 2.0 format (Start/End Index) for Extractive models (BERT). Generative LLMs need text-to-text pairs derived from the CSV. |

---

## 3. Exploratory Data Analysis & Strategy

Based on an analysis of `master_clauses.csv`, the following strategies apply to the fine-tuning process.

### A. Class Imbalance (Task 1)
* **Finding:** Risk clauses (e.g., Uncapped Liability) are sparse (<10% frequency).
* **Impact:** Model may converge to simply predicting "No Clause Found".
* **Strategy:**
    * **Prompt Engineering:** Use "Positive/Negative" instruction pairs. Include "No" examples to teach explicit absence detection.
    * **Oversampling:** Artificially increase frequency of rows containing rare risk clauses during epoch generation.

### B. Context Length (Hardware Constraints)
* **Finding:** Average length of "Text Context" + Instruction fits within 2048/4096 token limits.
* **Impact:** Confirms feasibility of QLoRA on consumer cards (RTX 3090/4090, T4/A100).
* **Strategy:** Truncation can be safely set at **2048 tokens** with minimal data loss.

### C. Output Formatting (Task 2)
* **Finding:** Answers occasionally contain multiple values separated by semicolons.
* **Impact:** Model must learn to parse this delimiter into a JSON array.
* **Strategy:** Instruction prompt must explicitly state: *"Return the result as a JSON object. If multiple entities exist, return them as a list of strings."*

---

## 4. Preprocessing Pipeline Summary

Steps to prepare `master_clauses.csv` for the model:

1. **Filter:** Remove metadata columns (Filename).
2. **Pair:** Map `[Category]` columns to `[Category] Answer` columns.
3. **Split:** Randomly select 10-15% of rows (contracts) for `validation.jsonl`.
4. **Format:** Convert remaining rows to JSONL format.
5. **Tokenize:** Apply tokenizer with padding to max_length (2048).

**JSONL Example:**
```json
{
  "instruction": "Identify the Expiration Date in the following clause and output as JSON.",
  "input": "The agreement shall remain in effect until... [Clause Text]",
  "output": "{\"Expiration Date\": \"12/31/2025\"}"
}
```
