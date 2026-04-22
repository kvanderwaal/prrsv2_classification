import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from lightgbm import LGBMClassifier
import shap
import argparse
import os

# Fixed classification threshold — no longer adjusted via precision-recall curve
THRESHOLD = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YoY growth rate prediction')
    parser.add_argument('-n', '--name', type=str, required=True, help="Name of quarter in YYYY_MMM format for example 2024_Mar")
    parser.add_argument('-t', '--traindat', type=str, required=True, help="Recent quarter training data in CSV format")
    parser.add_argument('-v', '--varsum', type=str, required=True, help="Recent quarter active variant summary in CSV format")
    parser.add_argument('-pp', '--prepred', type=str, required=True, help="All predictions updated til the previous quarter in CSV format")
    parser.add_argument('-pf', '--preperf', type=str, required=True, help="All prediction performace updated til the previous quarter in CSV format")
    parser.add_argument('-o', '--outdir', type=str, required=True, help="Output directory for the results")
    args = parser.parse_args()

    # Datetime of Tob (observation time point)
    name = args.name
    Tob = pd.to_datetime(name, format="%Y_%b")

    # load train data
    df0 = pd.read_csv(args.traindat, sep=',', header=0)

    # load variant summary
    dsum = pd.read_csv(args.varsum, sep=',', header=0)

    # load previous predictions
    dpp = pd.read_csv(args.prepred, sep=',', header=0)

    # load previous prediction performance
    dpf = pd.read_csv(args.preperf, sep=',', header=0)

    df = df0.loc[:,'gdist':'CAGR12_Class']
    target_column = 'CAGR12_Class'
    dtrain = df[df['CAGR12_Class'].isin(['high', 'low'])]
    dftest = df0[~df0['CAGR12_Class'].isin(['high', 'low'])]
    dtest = df[~df['CAGR12_Class'].isin(['high', 'low'])]
    class_mapping = {"low": 0, "high": 1}
    dtrain.loc[:, target_column] = dtrain[target_column].map(class_mapping)

    # Define features and target
    X_train = dtrain.drop("CAGR12_Class", axis=1)
    y_train = dtrain["CAGR12_Class"]
    y_train = y_train.astype(int)
    X_test = dtest.drop("CAGR12_Class", axis=1)

    # No SMOTE — class_weight="balanced" in LightGBM handles imbalance
    lgbm = LGBMClassifier(objective="binary", class_weight="balanced", random_state=42, verbosity=-1)
    param_dist = {
        "num_leaves": [31, 50, 70],
        "max_depth": [10, 20, 30, -1],
        "learning_rate": [0.1, 0.01, 0.05],
        "n_estimators": [100, 200, 500],
        "min_child_samples": [20, 50, 100]
    }

    # Set up 10-fold cross-validation with randomized search
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=lgbm, param_distributions=param_dist, n_iter=10, cv=kf, scoring="f1", random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Train the best model on the full training data
    best_model = random_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Predict on the test set using fixed threshold 0.5
    y_test_pred_proba = best_model.predict_proba(X_test)
    y_test_pred = (y_test_pred_proba[:, 1] >= THRESHOLD).astype(int)

    dftest2 = dftest.copy()
    dftest2['y_test_pred'] = y_test_pred

    dftest2 = dftest2.copy()
    for i, class_label in enumerate(best_model.classes_):
        dftest2[f'proba_class_{class_label}'] = y_test_pred_proba[:, i]

    # --- SHAP explainability for high-growth predictions ---
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)

    # For binary LightGBM, shap_values may be a list [class0, class1] or a 3D array.
    # Extract SHAP values for class 1 (high growth).
    if isinstance(shap_values, list):
        shap_high = shap_values[1]       # shape: (n_samples, n_features)
    elif shap_values.ndim == 3:
        shap_high = shap_values[:, :, 1] # shape: (n_samples, n_features)
    else:
        shap_high = shap_values          # already (n_samples, n_features) for binary

    # Build a SHAP dataframe aligned to X_test
    shap_df = pd.DataFrame(shap_high, columns=X_test.columns, index=X_test.index)
    shap_df['variant'] = dftest['variant'].values
    shap_df['y_test_pred'] = y_test_pred
    shap_df['Tob'] = pd.to_datetime(dftest['Tob'].values)

    # Keep only high-growth predictions for the current Tob
    shap_high_df = shap_df[
        (shap_df['y_test_pred'] == 1) &
        (shap_df['Tob'] == Tob)
    ].copy()

    feature_cols = X_test.columns.tolist()
    TOP_N = 10

    # --- Per-variant top-10 SHAP table (long format) ---
    # One row per (variant, rank): useful for programmatic use / downstream analysis
    per_variant_records = []
    for _, row in shap_high_df.iterrows():
        shap_scores = row[feature_cols].astype(float)
        ranked = shap_scores.sort_values(ascending=False).head(TOP_N)
        for rank, (feat, val) in enumerate(ranked.items(), start=1):
            per_variant_records.append({
                'Tob': name,
                'variant': row['variant'],
                'rank': rank,
                'feature': feat,
                'shap_value': round(val, 6)
            })

    shap_per_variant = pd.DataFrame(per_variant_records)

    # --- Quarter-level summary table (wide format) ---
    # Rows = rank 1–10, Columns = variants predicted high this quarter.
    # Each cell shows "feature (SHAP=X.XXX)" for readability.
    if not shap_per_variant.empty:
        shap_wide = shap_per_variant.pivot(index='rank', columns='variant', values='feature')
        shap_val_wide = shap_per_variant.pivot(index='rank', columns='variant', values='shap_value')

        # Combine feature name + SHAP value into a single readable cell
        shap_summary = pd.DataFrame(index=shap_wide.index, columns=shap_wide.columns)
        for col in shap_wide.columns:
            shap_summary[col] = shap_wide[col] + ' (' + shap_val_wide[col].round(3).astype(str) + ')'

        shap_summary.index.name = 'rank'
        shap_summary.columns.name = None
        shap_summary.insert(0, 'Tob', name)
    else:
        # No high-growth variants this quarter — write an empty placeholder
        shap_summary = pd.DataFrame({'Tob': [name], 'note': ['No high-growth variants predicted this quarter']})
        shap_per_variant = pd.DataFrame(columns=['Tob', 'variant', 'rank', 'feature', 'shap_value'])

    ### Evaluate previous predictive performance (tracking only — does not affect threshold)
    def evaluate_prediction_performance(df0, dpp):
        # Step 1: Prepare actual data
        df1 = df0[['Tob', 'variant', 'CAGR12_Class']].copy()
        df1['Tob'] = pd.to_datetime(df1['Tob'])
        df1['CAGR12_Class'] = df1['CAGR12_Class'].map({'low': 0, 'high': 1})
        df1_clean = df1.dropna(subset=['CAGR12_Class']).copy()

        # Step 2: Prepare predicted data
        dpp2 = dpp.copy()
        dpp2['Tob'] = pd.to_datetime(dpp2['Tob'])

        # Step 3: Merge on Tob and variant
        merged = pd.merge(dpp2, df1_clean, on=['Tob', 'variant'], how='inner')
        merged['CAGR12_Class'] = merged['CAGR12_Class'].astype(int)
        merged['y_test_pred'] = merged['y_test_pred'].astype(int)

        y_test = merged['CAGR12_Class']
        y_pred = merged['y_test_pred']
        y_pred_proba = merged['proba_class_1']

        def compute_metrics(y_test, y_pred, y_pred_proba):
            conf_matrix = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = conf_matrix.ravel()

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average="weighted", zero_division=0),
                'recall': recall_score(y_test, y_pred, average="weighted", zero_division=0),
                'f1': f1_score(y_test, y_pred, average="weighted", zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'sensitivity': tp / (tp + fn) if (tp + fn) else np.nan,
                'specificity': tn / (tn + fp) if (tn + fp) else np.nan,
                'ppv': tp / (tp + fp) if (tp + fp) else np.nan,
                'npv': tn / (tn + fn) if (tn + fn) else np.nan,
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn,
            }
            return metrics

        # First row: All Tob
        overall_metrics = compute_metrics(y_test, y_pred, y_pred_proba)

        # Second row: Most recent Tob
        most_recent_Tob = df1_clean['Tob'].max()
        latest_merged = merged[merged['Tob'] == most_recent_Tob]
        latest_metrics = compute_metrics(
            latest_merged['CAGR12_Class'],
            latest_merged['y_test_pred'],
            latest_merged['proba_class_1']
        )

        # Step 4: Format Tob labels and build metrics_df
        def format_tob_label(tob):
            return tob.strftime('%Y_%b')  # e.g., '2023_Jun'

        metrics_df = pd.DataFrame([overall_metrics, latest_metrics])
        metrics_df.insert(0, 'Tob', ['all_Tob_since_2023_Jun', format_tob_label(most_recent_Tob)])

        return metrics_df

    metrics_df = evaluate_prediction_performance(df0, dpp)

    # concat performance history
    alltob_value = metrics_df.iloc[0]['Tob']
    dpf2 = dpf[dpf['Tob'] != alltob_value]
    dpf_new = pd.concat([metrics_df.iloc[[0]], metrics_df.iloc[1:], dpf2], ignore_index=True)

    # Apply fixed threshold 0.5 (already applied above during prediction, kept here for clarity)
    dftest2["y_test_pred"] = (dftest2["proba_class_1"] >= THRESHOLD).astype(int)

    predout = dftest2[['Tob', 'variant', 'y_test_pred', 'proba_class_0', 'proba_class_1']]
    predout.loc[:, 'Tob'] = pd.to_datetime(predout['Tob'])
    newpredout = predout[predout['Tob'] == Tob]
    newpredout = newpredout.drop(columns=['Tob'])
    newpredout = newpredout.rename(columns={
        'variant': 'variant.id',
        'y_test_pred': 'predicted growth rate',
        'proba_class_0': 'prob_of_YoY<=15%',
        'proba_class_1': 'prob_of_YoY>15%'
    })
    newpredout['predicted growth rate'] = newpredout['predicted growth rate'].map({1: 'high', 0: 'low'})
    dsumtest = reduce(lambda left, right: pd.merge(left, right, on=['variant.id'], how='left'), [dsum, newpredout])

    predout2 = predout[predout['Tob'] == Tob]
    predout2 = predout2.copy()
    predout2['threshold'] = THRESHOLD

    # add new prediction to cumulative history
    allprednow = pd.concat([dpp, predout2], axis=0)

    # Output file paths
    outfile2 = os.path.join(args.outdir, f"{args.name}_allpred.csv")
    outfile3 = os.path.join(args.outdir, f"{args.name}_predperf.csv")
    outfile_shap_long = os.path.join(args.outdir, f"{args.name}_shap_top10_per_variant.csv")
    outfile_shap_wide = os.path.join(args.outdir, f"{args.name}_shap_top10_summary.csv")

    # Write all outputs
    dsumtest.to_csv("variant.summary.active.pred.csv", sep=',', header=True, index=False, na_rep='NA')
    allprednow.to_csv(outfile2, sep=',', header=True, index=False, na_rep='NA')
    dpf_new.to_csv(outfile3, sep=',', header=True, index=False, na_rep='NA')
    shap_per_variant.to_csv(outfile_shap_long, sep=',', header=True, index=False, na_rep='NA')
    shap_summary.to_csv(outfile_shap_wide, sep=',', header=True, index=True, na_rep='NA')

    print(f"SHAP outputs written:")
    print(f"  Per-variant long format : {outfile_shap_long}")
    print(f"  Quarter summary (wide)  : {outfile_shap_wide}")
