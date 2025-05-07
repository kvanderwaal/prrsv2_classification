import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import argparse
import os

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

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X_train = X_resampled
    y_train = y_resampled

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

    # Predict on the test set
    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)
    dftest2 = dftest.copy()
    dftest2['y_test_pred'] = y_test_pred

    dftest2 = dftest2.copy()
    for i, class_label in enumerate(best_model.classes_):
        dftest2[f'proba_class_{class_label}'] = y_test_pred_proba[:, i]

    ### Check previous predictive performance and set new predict threshold
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

        # Step 5: Optimal threshold (on all Tob)
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        optimal_threshold = thresholds[np.argmax(f1_scores)]

        return metrics_df, optimal_threshold

    metrics_df, opt_thresh = evaluate_prediction_performance(df0, dpp)

    # concat performance history
    alltob_value = metrics_df.iloc[0]['Tob']
    dpf2 = dpf[dpf['Tob'] != alltob_value]
    dpf_new = pd.concat([metrics_df.iloc[[0]], metrics_df.iloc[1:], dpf2], ignore_index=True)

    #set new threshold
    dftest2["y_test_pred"] = (dftest2["proba_class_1"] > opt_thresh).astype(int)

    predout = dftest2[['Tob', 'variant', 'y_test_pred', 'proba_class_0', 'proba_class_1']]
    predout.loc[:, 'Tob'] = pd.to_datetime(predout['Tob'])
    newpredout = predout[predout['Tob'] == Tob]
    newpredout = newpredout.drop(columns=['Tob'])
    newpredout = newpredout.rename(columns={
        'variant': 'variant.id',
        'y_test_pred': 'predicted growth rate',
        'proba_class_0': 'prob_of_YoY<=20%',
        'proba_class_1': 'prob_of_YoY>20%'
    })
    newpredout['predicted growth rate'] = newpredout['predicted growth rate'].map({1: 'high', 0: 'low'})
    dsumtest = reduce(lambda left, right: pd.merge(left, right, on=['variant.id'], how='left'), [dsum, newpredout])

    predout2 = predout[predout['Tob'] == Tob]
    predout2 = predout2.copy()
    predout2['threshold'] = opt_thresh

    # add new prediction
    allprednow = pd.concat([dpp, predout2], axis=0)

    outfile1 = os.path.join(args.outdir, f"variant.summary.active.pred.csv")
    outfile2 = os.path.join(args.outdir, f"{args.name}_allpred.csv")
    outfile3 = os.path.join(args.outdir, f"{args.name}_predperf.csv")

    dsumtest.to_csv(outfile1, sep=',', header=True, index=False, na_rep='NA')
    allprednow.to_csv(outfile2, sep=',', header=True, index=False, na_rep='NA')
    dpf_new.to_csv(outfile3, sep=',', header=True, index=False, na_rep='NA')
