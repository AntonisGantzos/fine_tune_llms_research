# Legal Domain LLM Fine-Tuning Research

## Project Overview
This research focuses on fine-tuning Large Language Models (LLMs) for the **Legal Domain**. The primary objective is to adapt general-purpose models (such as Llama 3 or Mistral) to recognize, classify, and extract specific elements from complex legal documents with high precision.

## Problem Statement
Legal documents effectively require a high degree of distinct comprehension that general LLMs often lack. The specific challenges addressed in this research include:
*   **Unstructured Data:** Legal contracts contain critical data (dates, parties, clauses) buried in dense text.
*   **Strict Formatting Requirements:** Downstream systems often require output in valid JSON, which standard models often fail to produce consistently.
*   **Hallucinations:** In a legal context, factual accuracy is paramount. A model cannot invent clauses or dates.
*   **Hardware Constraints:** Fine-tuning massive models typically requires enterprise-grade hardware (A100s). This research aims to make this process accessible on consumer hardware.

## Research Goals
The goal of this project is to produce a fine-tuned model capable of performing three specific tasks using efficient training techniques.

### Core Tasks
1.  **Risk Clause Recognition (T1):** 
    *   *Goal:* Identify and list specific risk clauses within a contract (e.g., "Termination for Convenience," "Uncapped Liability").
    *   *Dataset:* CUAD.
2.  **Structured Entity Extraction (T2):**
    *   *Goal:* Extract entities like "Agreement Date," "Parties," and "Renewal Term" and output them strictly as a valid JSON object.
    *   *Dataset:* CUAD.
3.  **Jurisdiction & Governing Law Identification (T3):**
    *   *Goal:* Classify the governing law or jurisdiction of a given contract clause.
    *   *Dataset:* LEDGAR.

## Datasets

The training and evaluation data are derived from two primary sources:

### 1. LEDGAR (Labeled EDGAR)
*   **Purpose:** Used for contract provision classification (Task 3).
*   **Description:** A large-scale multi-label corpus consisting of approximately 850,000 contract provisions (paragraphs) scraped from SEC filings.
*   **Source:** Provisions are labeled with over 12,000 categories.
*   **Reference:** *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts* (LREC 2020) by Tuggener et al.

### 2. CUAD (Contract Understanding Atticus Dataset)
*   **Purpose:** Used for entity extraction and clause recognition (Task 1 & 2).
*   **Description:** An expert-annotated NLP dataset for legal contract review. It focuses on identifying specific commercial clauses (e.g., "Termination for Convenience", "Rofr/Rofo") and entities.
*   **Note:** While early experiments may reference the "Atticus Open Contract Dataset (AOK) Beta" (Oct 2020), this research targets the finalized **CUAD v1** (March 2021) standard.
*   **Reference:** *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review* (NeurIPS 2021) by The Atticus Project.

## Methodology & Techniques

To achieve these goals on consumer-grade hardware (e.g., Google Colab free tier, limited VRAM), this research utilizes Parameter-Efficient Fine-Tuning (PEFT):

*   **LoRA (Low-Rank Adaptation):** Used to introduce trainable rank decomposition matrices, allowing the pre-trained model to be frozen while adapting to legal syntax.
*   **QLoRA (Quantized LoRA):** A practical necessity for this research. By loading models in 4-bit quantization, memory usage for a Llama-3-8B model drops from ~16GB to ~6GB VRAM, making fine-tuning feasible without high-end GPUs.
*   **Prompt Engineering:** integrating specific instructions directly into the training data to guide model behavior.

## Evaluation Metrics

To ensure the model is practical and accurate, the following metrics are tracked:

*   **JSON Validity Score:** specifically for Task 2, measuring the percentage of generations that successfully parse as valid JSON code.
*   **Precision:** To measure hallucination rates (ensuring extracted text actually exists in the source).
*   **Exact Match (EM):** For rigid entities (e.g., Dates, State names).
*   **F1 Score:** For longer form extraction (e.g., Clauses) to allow for partial credit where the semantic meaning is captured even if the boundary is slightly off.