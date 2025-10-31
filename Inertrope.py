import re
import argparse
import pandas as pd
import numpy as np
import joblib
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


# ==========================================================
#  Helper Functions
# ==========================================================
def detect_sample_column(df):
    """Auto-detect a sample ID column (case/format insensitive)."""
    for col in df.columns:
        normalized = re.sub(r'[^a-z0-9]', '', col.lower())
        normalized = normalized.rstrip('s')
        if normalized in {"sample", "sampleid", "sid", "id"}:
            print(f"[INFO] Detected sample identifier column: '{col}'")
            return col
    raise ValueError(
        "No valid sample ID column found (expected: 'Sample_ID', 'sample', 'Sample Id', 'Samples', etc.)."
    )


def normalize_data(df, lognorm=True, scaling="none"):
    """Applies log normalization and optional scaling to numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    meta_cols = df.columns.difference(numeric_cols)
    data = df[numeric_cols].copy().astype(float)

    if lognorm:
        nonzero = np.abs(data.values[np.nonzero(data.values)])
        if nonzero.size == 0:
            raise ValueError("No nonzero values found for log normalization.")
        min_val = np.min(nonzero) / 10.0
        eps = 1e-12

        for col in data.columns:
            x = data[col].values
            data[col] = np.log10((x + np.sqrt(x ** 2 + min_val ** 2 + eps)) / 2.0)

        if np.isinf(data.values).any():
            finite_min = np.nanmin(data.values[np.isfinite(data.values)])
            data.replace([-np.inf, np.inf], finite_min, inplace=True)
    else:
        min_val = 0.0

    s = scaling.lower()
    if s == "autoscale":
        data = (data - data.mean()) / data.std(ddof=1)
    elif s == "pareto":
        data = (data - data.mean()) / np.sqrt(data.std(ddof=1))

    df_out = pd.concat([df[meta_cols], data], axis=1)
    df_out.fillna(0, inplace=True)
    return df_out, min_val


def align_features(model, features_df):
    """Aligns DataFrame columns to match model features."""
    print("[INFO] Aligning features...")
    try:
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
        elif hasattr(model, 'feature_names_'):
            model_features = model.feature_names_
        elif hasattr(model, 'n_features_in_'):
            model_features = [f'feature_{i}' for i in range(model.n_features_in_)]
            if len(model_features) == features_df.shape[1]:
                print("[WARN] Model feature names not found. Assuming column order is correct.")
                features_df.columns = model_features
            else:
                print(f"[ERROR] Model expects {model.n_features_in_} features, but data has {features_df.shape[1]}.")
                return None
        else:
            print("[ERROR] Could not determine model's expected feature names.")
            return None

        # Reindex with zero fill for missing columns
        final_features_df = features_df.reindex(columns=model_features, fill_value=0)
        print(f"[INFO] Using {final_features_df.shape[1]} aligned features for prediction.")
        return final_features_df
    except Exception as e:
        print(f"[ERROR] Feature alignment failed: {e}")
        return None


def melt_and_extract_tsfresh(data, time_prefix="Time_"):
    """Extracts tsfresh features from Time_ columns (robust to any index name)."""
    time_cols = [c for c in data.columns if c.startswith(time_prefix)]
    if not time_cols:
        raise ValueError(f"No columns found with prefix '{time_prefix}'")

    # Reset index safely and rename first column to 'id'
    df_reset = data.reset_index()
    id_col_name = df_reset.columns[0]
    df_reset = df_reset.rename(columns={id_col_name: "id"})

    df_long = df_reset.melt(
        id_vars=["id"],
        value_vars=time_cols,
        var_name="time",
        value_name="value"
    )
    df_long["time"] = df_long["time"].str.replace(time_prefix, "").astype(float)
    print(f"[INFO] Converted to long-format: {df_long.shape}")

    features = extract_features(
        df_long,
        column_id="id",
        column_sort="time",
        column_value="value",
        disable_progressbar=False
    )
    impute(features)
    print(f"[INFO] tsfresh features extracted: {features.shape}")
    return features


def predict_multiclass(features, model_path):
    """Predicts class labels and probabilities using ExtraTrees model."""
    model = joblib.load(model_path)
    print(f"[INFO] Model loaded from {model_path}")

    final_features = align_features(model, features)
    if final_features is None:
        raise ValueError("Feature alignment failed.")

    if final_features.isna().any().any():
        print("[WARN] NaN values detected in features before prediction. Filling with 0.")
        final_features = final_features.fillna(0)

    preds = model.predict(final_features)
    probs = model.predict_proba(final_features)

    class_map = {0: "Healthy", 1: "Benign", 2: "Cancer"}
    pred_labels = pd.Series(preds).map(class_map)

    prob_df = pd.DataFrame(
        probs,
        columns=["Prob_Healthy", "Prob_Benign", "Prob_Cancer"]
    )

    # output structure
    result_df = pd.DataFrame({
        "Sample_ID": features.index,
        "Predicted_Status": pred_labels.values
    })

    full_result = pd.concat([result_df, prob_df], axis=1)
    print(f"[SUCCESS] Predictions complete for {len(features)} samples.")
    return full_result


# ==========================================================
# 🧩 Class inertrope
# ==========================================================
class inertrope:
    """Main namespace class for Inertropy ITC pipelines."""

    TIME_PREFIX = "Time_"

    # Default model paths (reconfigurable)
    MODELS = {
        "ITC": "/storage/savi/saveenas/Projects/Inertropy/shiva_new/latest/final_extratrees_itc.joblib",
        "COMBINED": "/storage/savi/saveenas/Projects/Inertropy/shiva_new/latest/combined/final_extratrees_combined.joblib"
    }

    # ------------------------------------------------------
    @classmethod
    def configure_models(cls, itc_model_path=None, combined_model_path=None):
        """Reconfigure model paths at runtime."""
        if itc_model_path:
            cls.MODELS["ITC"] = itc_model_path
            print(f"[CONFIG] ITC model path set to: {cls.MODELS['ITC']}")
        if combined_model_path:
            cls.MODELS["COMBINED"] = combined_model_path
            print(f"[CONFIG] Combined model path set to: {cls.MODELS['COMBINED']}")

    # ------------------------------------------------------
    @staticmethod
    def pred_itc(df_or_path, out_csv="itc_predictions.csv"):
        """Pipeline for pure time-series ITC data."""
        print(f"[INFO] Running inertrope.pred_itc (time-series only)")

        df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path
        sample_col = detect_sample_column(df)
        sample_ids = df[sample_col].copy()

        time_cols = [c for c in df.columns if c.startswith(inertrope.TIME_PREFIX)]
        if not time_cols:
            raise ValueError("No 'Time_' columns found.")

        df_time = df[time_cols].copy()
        df_time_norm, _ = normalize_data(df_time, lognorm=True)

        # Safe concatenation to prevent fragmentation
        df_time_norm = pd.concat([sample_ids, df_time_norm], axis=1).set_index(sample_col)

        features = melt_and_extract_tsfresh(df_time_norm, time_prefix=inertrope.TIME_PREFIX)
        results = predict_multiclass(features, inertrope.MODELS["ITC"])

        results.to_csv(out_csv, index=False)
        print(f"[DONE] inertrope.pred_itc results saved → {out_csv}")
        return results

    # ------------------------------------------------------
    @staticmethod
    def pred_combined(df_or_path, out_csv="itc_predictions_combined.csv"):
        """Pipeline for hybrid time + static wavelength datasets."""
        print(f"[INFO] Running inertrope.pred_combined (time + static)")

        df = pd.read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path
        sample_col = detect_sample_column(df)
        sample_ids = df[sample_col].copy()

        time_cols = [c for c in df.columns if c.startswith(inertrope.TIME_PREFIX)]
        static_cols = [c for c in df.columns if not c.startswith(inertrope.TIME_PREFIX)
                       and df[c].dtype != "object" and c != sample_col]

        df_time = df[time_cols].copy()
        df_time_norm, _ = normalize_data(df_time, lognorm=True)
        df_time_norm = pd.concat([sample_ids, df_time_norm], axis=1).set_index(sample_col)

        df_tsfresh = melt_and_extract_tsfresh(df_time_norm, time_prefix=inertrope.TIME_PREFIX)
        df_static = df[[sample_col] + static_cols].set_index(sample_col) if static_cols else pd.DataFrame(index=sample_ids)

        df_combined = pd.concat([df_tsfresh, df_static], axis=1)
        df_combined = df_combined.loc[~df_combined.index.duplicated(keep='first')]

        results = predict_multiclass(df_combined, inertrope.MODELS["COMBINED"])
        results.to_csv(out_csv, index=False)
        print(f"[DONE] inertrope.pred_combined results saved → {out_csv}")
        return results
