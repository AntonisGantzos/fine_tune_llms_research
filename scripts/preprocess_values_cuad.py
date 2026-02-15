import pandas as pd
import re
import os
import csv 

def clean_text(text):
    """
    1. Converts input to string.
    2. Removes all characters that are NOT alphanumeric (a-z, A-Z, 0-9) or whitespace.
    3. Normalizes whitespace (removes newlines/tabs, converts multiple spaces to single space).
    """
    if pd.isna(text):
        return ""
    
    # Ensure it's a string
    text = str(text)
    
    # Remove special characters (keep only letters, numbers, and spaces)
    # If you want to keep punctuation like periods/commas, change pattern to r'[^a-zA-Z0-9\s.,]'
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Replace newlines and tabs with a single space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_dataset(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    print(f"Loading data from {input_path}...")
    data = []
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        # Use csv.Sniffer to deduce format if possible, or enforce standard strictness
        reader = csv.DictReader(f) 
        for i, row in enumerate(reader):
            data.append(row)

    # Convert the list of dicts to a DataFrame
    df = pd.DataFrame(data)

    print("Cleaning all text columns...")
    # Apply cleaning to the entire DataFrame
    # map(clean_text) applies the function to every cell in the DataFrame
    df_cleaned = df.map(clean_text)

    # Optional: Filter out rows that became completely empty
    # df_cleaned = df_cleaned[df_cleaned.astype(bool).any(axis=1)]

    print(f"Saving cleaned data to {output_path}...")
    df_cleaned.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    # Adjust paths relative to your workspace root
    INPUT_FILE = "data/cuad/master_clauses.csv"
    OUTPUT_FILE = "data/cuad/master_clauses_cleaned.csv"
    try:
        process_dataset(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"An error occurred: {e}")