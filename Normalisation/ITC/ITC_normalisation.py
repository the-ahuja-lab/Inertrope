import pandas as pd
import re


# ============================================================
# 1. Read raw data and validate numeric columns
# ============================================================

df = pd.read_csv("ITC_data_raw.csv")

# Work on a copy so the original raw data remains unchanged
corrected_data = df.copy()

non_numeric_cols = []

for col in corrected_data.columns:
    if col == "Time":
        continue

    try:
        corrected_data[col] = pd.to_numeric(
            corrected_data[col],
            errors="raise"
        )
    except (ValueError, TypeError):
        non_numeric_cols.append(col)

print(
    f"\nColumns containing non-numeric or mixed data: "
    f"{len(non_numeric_cols)}"
)

if non_numeric_cols:
    print(non_numeric_cols[:15])
    raise ValueError(
        "Non-numeric values were found. Clean these columns "
        "before applying corrections."
    )


# ============================================================
# 2. Read and validate correction factors
# ============================================================

corrections_df = pd.read_csv(
    "Raw plasma data - corrections(1).csv"
)

required_correction_cols = {"Sample Id", "Correction"}

missing_correction_cols = (
    required_correction_cols - set(corrections_df.columns)
)

if missing_correction_cols:
    raise KeyError(
        f"Missing correction columns: {missing_correction_cols}"
    )

corrections_df["Sample Id"] = (
    corrections_df["Sample Id"]
    .astype(str)
    .str.strip()
)

corrections_df["Correction"] = pd.to_numeric(
    corrections_df["Correction"],
    errors="raise"
)

# Prevent ambiguous correction factors
duplicate_ids = corrections_df.loc[
    corrections_df["Sample Id"].duplicated(keep=False),
    "Sample Id"
].unique()

if len(duplicate_ids) > 0:
    raise ValueError(
        f"Duplicate correction factors found for: "
        f"{duplicate_ids.tolist()}"
    )

corrections_map = corrections_df.set_index(
    "Sample Id"
)["Correction"].to_dict()


# ============================================================
# 3. Multiply sample-specific columns by correction factors
# ============================================================

correction_summary = []

for sample_id, factor in corrections_map.items():

    # Exact underscore-delimited token matching.
    # PB1 matches C7_PB1_1 and B_PB1,
    # but does not match C7_PB10_1.
    pattern = re.compile(
        rf"(^|_){re.escape(sample_id)}(_|$)"
    )

    matching_cols = [
        col
        for col in corrected_data.columns
        if col != "Time" and pattern.search(col)
    ]

    if not matching_cols:
        correction_summary.append({
            "Sample ID": sample_id,
            "Correction Factor": factor,
            "Corrected Columns": "",
            "Status": "No matching columns"
        })
        continue

    corrected_data.loc[:, matching_cols] = (
        corrected_data[matching_cols].mul(factor)
    )

    correction_summary.append({
        "Sample ID": sample_id,
        "Correction Factor": factor,
        "Corrected Columns": ", ".join(matching_cols),
        "Status": "Corrected"
    })

correction_summary_df = pd.DataFrame(correction_summary)


# ============================================================
# 4. Resolve the two common-control columns
# ============================================================

# This supports both the names stated in your description
# and the names used in your original code.
control_aliases = {
    "BB": ["BB", "B_B"],
    "B_C7": ["B_C7", "C7_buffer"]
}


def resolve_column(dataframe, aliases, control_name):
    """Return the first available column from a list of aliases."""
    for alias in aliases:
        if alias in dataframe.columns:
            return alias

    raise KeyError(
        f"Could not find the {control_name} control column. "
        f"Expected one of: {aliases}"
    )


bb_col = resolve_column(
    corrected_data,
    control_aliases["BB"],
    "BB"
)

b_c7_col = resolve_column(
    corrected_data,
    control_aliases["B_C7"],
    "B_C7"
)

print(f"BB control column: {bb_col}")
print(f"B_C7 control column: {b_c7_col}")


# ============================================================
# 5. Plasma-buffer and common-control normalization
# ============================================================

normalized_df = corrected_data[["Time"]].copy()

normalization_summary = []

# Matches:
# C7_PN1_1
# C7_PB10_2
# C7_PC3_1
sample_pattern = re.compile(
    r"^C7_P([NBC])(\d+)_(\d+)$"
)

for sample_col in corrected_data.columns:

    match = sample_pattern.fullmatch(sample_col)

    if not match:
        continue

    sample_group = match.group(1)    # B, N or C
    sample_number = match.group(2)   # 1, 2, 10, ...
    replicate = match.group(3)       # 1, 2, ...

    plasma_buffer_col = (
        f"B_P{sample_group}{sample_number}"
    )

    if plasma_buffer_col not in corrected_data.columns:
        print(
            f"Skipping {sample_col}: "
            f"missing plasma-buffer column "
            f"{plasma_buffer_col}"
        )
        continue

    output_col = (
        f"Norm_C7_P{sample_group}"
        f"{sample_number}_{replicate}"
    )

    # Step 1: subtract the matching plasma buffer
    plasma_normalized = (
        corrected_data[sample_col]
        - corrected_data[plasma_buffer_col]
    )

    # Step 2: subtract both common controls
    normalized_df[output_col] = (
        plasma_normalized
        - corrected_data[bb_col]
        - corrected_data[b_c7_col]
    )

    normalization_summary.append({
        "Normalized Column": output_col,
        "Sample Column": sample_col,
        "Plasma Buffer Column": plasma_buffer_col,
        "BB Column": bb_col,
        "B_C7 Column": b_c7_col,
        "Calculation": (
            f"({sample_col} - {plasma_buffer_col}) "
            f"- {bb_col} - {b_c7_col}"
        )
    })

normalization_summary_df = pd.DataFrame(
    normalization_summary
)


# ============================================================
# 6. Arrange columns as PB, PN and PC
# ============================================================

group_order = {
    "B": 0,
    "N": 1,
    "C": 2
}


def normalized_column_sort_key(col):
    """
    Sort by:
    1. PB, PN, PC
    2. sample number
    3. replicate number
    """
    match = re.fullmatch(
        r"Norm_C7_P([BNC])(\d+)_(\d+)",
        col
    )

    if not match:
        return (99, 99, 99, col)

    sample_group = match.group(1)
    sample_number = int(match.group(2))
    replicate = int(match.group(3))

    return (
        group_order[sample_group],
        sample_number,
        replicate,
        col
    )


normalized_cols = [
    col
    for col in normalized_df.columns
    if col != "Time"
]

normalized_cols = sorted(
    normalized_cols,
    key=normalized_column_sort_key
)

normalized_df = normalized_df[
    ["Time"] + normalized_cols
]
normalized_df
