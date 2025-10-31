import pandas as pd
import numpy as np
import re

df = pd.read_csv("ITC_data_raw.csv")

#Check for nan
non_numeric_cols = []

for c in df.columns:
    if c == "Time":
        continue
    # try converting to numeric; if any fail → text present
    try:
        pd.to_numeric(df[c], errors='raise')
    except Exception:
        non_numeric_cols.append(c)

print(f"\n Columns containing text / mixed data: {len(non_numeric_cols)}")
if non_numeric_cols:
    print(non_numeric_cols[:15])  # show first few



#read correction factor
corrections_df = pd.read_csv("Raw plasma data - corrections(1).csv")



# Create a simple correction map: {"PB1": value, "PB2": value, ...}
corrections_map = dict(zip(corrections_df["Sample Id"], corrections_df["Correction"]))

# Create a list to store summary info
correction_summary = []

# Apply corrections
for sample_id, factor in corrections_map.items():
    # Find columns that contain the sample ID
    pattern = rf"(?:^|)({sample_id})(?:|$)"
    matching_cols = [col for col in corrected_data.columns if re.search(pattern, col)]

    if matching_cols:
        corrected_data[matching_cols] = corrected_data[matching_cols].apply(lambda col: col * factor)

        # Store which columns were corrected and with what factor
        correction_summary.append({
            "Sample ID": sample_id,
            "Correction Factor": factor,
            "Corrected Columns": matching_cols
        })


# Extract the two columns into a new DataFrame
new_df = corrected_data[['B_B', 'C7_buffer']].copy()


# Create new DataFrame to store normalized values (plasma-control normalization)
normalized_df = pd.DataFrame()
normalized_df["Time"] = corrected_data["Time"]

# Store summary info
normalization_summary = []

# Loop through all columns
for col in corrected_data.columns:
    # Match columns like C7_PN1_1, C7_PB1_1, C7_PC1_1
    match = re.match(r'C7_P([NBC])(\d+)_(\d+)', col)
    if match:
        letter = match.group(1)   # N, B, or C
        pc = match.group(2)       # Number
        rep = match.group(3)      # Replicate

        # Use the same letter as the original column for baseline
        b_col = f'B_P{letter}{pc}'
        norm_col = f'Norm_C7_P{letter}{pc}_{rep}'

        # Perform normalization
        normalized_df[norm_col] = corrected_data[col] / corrected_data[b_col]

        # Record in summary
        normalization_summary.append({
            "Normalized Column": norm_col,
            "Original Column": col,
            "Baseline Column": b_col
        })



# Subtract BB from each row of each column in normalized_df, excluding "Time"
for col in normalized_df.columns:
    if col != "Time":
        normalized_df[col] = normalized_df[col] / new_df["B_B"]


# Subtract BB from each row of each column in normalized_df, excluding "Time"
for col in normalized_df.columns:
    if col != "Time":
        normalized_df[col] = normalized_df[col] / new_df["C7_buffer"]

#Rearrange df
# Keep Time first
time_col = ['Time']

# Get all other columns
other_cols = [c for c in normalized_df.columns if c != 'Time']

# Initialize group dictionary
grouped_cols = {'PB': [], 'PN': [], 'PC': [], 'other': []}

# Loop through columns safely
for col in other_cols:
    m = re.match(r'.*(P[BPNC]\d+)\d+', col)  # match PB, PN, PC pattern
    if m:
        # Extract PB / PN / PC and number
        letter_num = m.group(1)  # e.g., PB10
        letter = letter_num[:2]  # PB / PN / PC
        num = int(letter_num[2:])  # 10
        if letter in grouped_cols:
            grouped_cols[letter].append((num, col))
        else:
            grouped_cols['other'].append(col)
    else:
        grouped_cols['other'].append(col)

# Flatten grouped columns
sorted_cols = time_col
for t in ['PB', 'PN', 'PC']:
    sorted_cols.extend([col for num, col in sorted(grouped_cols[t], key=lambda x: x[0])])

# Add other columns
sorted_cols.extend(grouped_cols['other'])

# Reorder DataFrame
normalized_df = normalized_df[sorted_cols]
normalized_df
