# CUAD Notebook — Cell-by-Cell Analysis

Notebook: `CUAD_dataset_exploration.ipynb`

Below is a concise description of each cell (by cell number) and how it fits into the dataset processing / fine-tuning pipeline.

- **Cell 1 — Markdown:** Introduction and high-level summary of CUAD, tasks (T1/T2/T3), and dataset stats.  
  - Purpose: Context for the notebook.  
  - Pipeline role: Explains research goals and reasons for the analysis.

- **Cell 2 — Code (imports):** Imports libraries (`datasets`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `json`, `transformers`, etc.).  
  - Purpose: Prepare runtime environment.  
  - Pipeline role: Dependencies required for loading, analyzing, tokenizing, and plotting data.

- **Cell 3 — Code (load_dataset & split):** Attempts to load CUAD via `datasets.load_dataset`, creates a train/test split if only `train` exists.  
  - Purpose: Obtain canonical CUAD from Hugging Face or create a local split.  
  - Pipeline role: Source data ingestion; ensures a reproducible train/test split when dataset lacks explicit test set.

- **Cell 4 — Code (peek sample):** Prints `cuad['train'][0]` to inspect a single example structure.  
  - Purpose: Quick sanity-check of dataset item format.  
  - Pipeline role: Confirms fields available (pdf, paragraphs, qas) before parsing.

- **Cell 5 — Markdown:** Notes about exploring embedded PDF content and `pdfplumber` usage.  
  - Purpose: Explains approach to extract contract text from PDF objects.  
  - Pipeline role: Documents how raw text is retrieved when using the HF-provided PDF objects.

- **Cell 6 — Code (pdf object extraction):** Accesses `pdf` object from a sample, prints metadata, extracts page text and tables.  
  - Purpose: Demonstrates extracting raw contract text/tables from PDF entry.  
  - Pipeline role: Provides example of low-level extraction used if raw clause text is required or to validate CSV extraction fidelity.

- **Cell 7 — Code (load local CUAD_v1.json):** Attempts to open `data/cuad/CUAD_v1.json` and preview the `data` structure, paragraphs, and Q/A entries.  
  - Purpose: Read local JSON format of CUAD and inspect schema.  
  - Pipeline role: Verifies presence and structure of local dataset files for downstream CSV conversion and EDA.

- **Cell 8 — Markdown:** Section header for dataset structure & schema exploration.  
  - Purpose: Signpost for next analysis steps.  
  - Pipeline role: Organizes the notebook for clear workflow progression.

- **Cell 9 — Markdown:** Explains mapping of 41 clause categories to Tasks (T1 vs T2).  
  - Purpose: Clarifies which categories map to classification vs extraction tasks.  
  - Pipeline role: Guides preprocessing and labeling decisions for training data formatting.

- **Cell 10 — Code (task category counts):** Defines `task_2_categories`, iterates `cuad_data` to count answered annotations and separate T1/T2 examples.  
  - Purpose: Quantify examples per task type.  
  - Pipeline role: Helps estimate dataset size per task, informs class-balance and sampling strategies.

- **Cell 11 — Markdown:** Notes on `master_clauses.csv` column roles (Context vs Answer) and how to interpret columns.  
  - Purpose: Explain CSV structure for upcoming CSV read.  
  - Pipeline role: Prepares reader for master CSV mapping -> core training source.

- **Cell 12 — Code (load master_clauses.csv):** Reads `data/cuad/master_clauses.csv` using `csv.DictReader` into a DataFrame `df`.  
  - Purpose: Robust CSV ingestion to handle inconsistencies.  
  - Pipeline role: Primary ingestion step when working from the provided CSV rather than JSON/PDF.

- **Cell 13 — Code (`df.head(3)`):** Displays the first few rows for a quick sanity check.  
  - Purpose: Inspect dataframe columns and a few examples.  
  - Pipeline role: Quick validation of read step before cleaning.

- **Cell 14 — Code (clean_text & clean column names):** Defines a `clean_text` helper and normalizes column names by stripping special chars.  
  - Purpose: Normalize headers and prepare safe column names.  
  - Pipeline role: Prevents parsing errors and standardizes column keys for mapping context/answer pairs.

- **Cell 15 — Code (rename answer columns):** Builds `new_columns` mapping to append `_Answer` suffix to Answer columns and renames them.  
  - Purpose: Standardize answer-column names for predictable access.  
  - Pipeline role: Simplifies later loops that pair `Category` with `Category_Answer`.

- **Cell 16 — Code (print columns):** Prints column names after renaming for verification.  
  - Purpose: Confirm rename succeeded.  
  - Pipeline role: Verification step for schema cleanup.

- **Cell 17 — Code (extract categories):** Scans columns to identify category/context columns paired with `_Answer` columns and collects `categories`.  
  - Purpose: Build list of clause types present in the CSV.  
  - Pipeline role: Produces canonical list of training labels to iterate over for EDA and conversion to JSONL.

- **Cell 18 — Code (counts check):** Shows counts: `len(all_columns)` and `len(categories)` to validate mapping coverage.  
  - Purpose: Quick numeric check on columns vs detected categories.  
  - Pipeline role: Sanity-check prior to per-category analysis.

- **Cell 19 — Code (answer distribution analysis):** Identifies all `_Answer` columns, computes distribution (Empty/No/Valid Text) and plots pie charts per category.  
  - Purpose: Visualize presence/absence of answers per category.  
  - Pipeline role: EDA to reveal sparsity and label distribution; informs oversampling and class-weighting strategies.

- **Cell 20 — Markdown:** Human-readable grouping of risk-related categories (legal concepts list).  
  - Purpose: Group similar clauses under legal concepts for interpretation.  
  - Pipeline role: Helps prioritize which clauses are high-risk and may need special handling during training.

- **Cell 21 — Code (Task 1 analysis):** Defines `risk_categories`, computes counts/percentages of contracts containing each clause, and plots a bar chart of sparsity.  
  - Purpose: Quantify sparsity for Task 1 risk clauses.  
  - Pipeline role: Directly informs imbalance mitigation (oversample, weighted loss, prompt design).

- **Cell 22 — Code (Task 2 analysis):** Defines `entity_categories` (e.g., `Agreement Date`, `Parties`), samples non-empty answers and prints examples for normalization inspection.  
  - Purpose: Peek at actual answer strings to check normalization needs (dates, semicolons, lists).  
  - Pipeline role: Guides preprocessing rules (date parsing, splitting multi-values) and JSON-output templates.

- **Cell 23 — Code (context length analysis):** Estimates token-like lengths for clause contexts, plots distribution, computes average and max estimated tokens.  
  - Purpose: Ensure clause contexts fit LLM context windows (recommendation: 2048).  
  - Pipeline role: Sets tokenizer/truncation `max_length` and batching strategy for fine-tuning (avoid truncating essential evidence).

- **Cell 24 — Code (train/validation split):** Splits `df` into `train_df` and `val_df` using `train_test_split(test_size=0.15)` and prints counts.  
  - Purpose: Create contract-level train/validation split.  
  - Pipeline role: Prevents leakage; saves split for training/validation cycles.

- **Cell 25 — Code (instruction tuning preview):** Constructs an instruction-format example (`instruction`, `input`, `output`) using `create_prompt_example` and prints a sample JSON entry (Task 2 example).  
  - Purpose: Demonstrate final JSONL/Instruction-Tuning format expected by Llama-style SFT/LoRA pipeline.  
  - Pipeline role: Template for converting cleaned CSV rows into JSONL training examples consumed by fine-tuning scripts.

---

Notes / Next steps (practical):
- The notebook is structured to accept either HF `datasets` objects (PDF/JSON) or a local `master_clauses.csv`.  
- The canonical pipeline implied by the notebook is: ingest (PDF/JSON/CSV) → normalize column names & text → identify category pairs → EDA (sparsity, token length, samples) → contract-level split → export JSONL instruction examples → tokenizer + LoRA/PEFT training.

File created: [CUAD_notebook_cell_analysis.md](CUAD_notebook_cell_analysis.md)
