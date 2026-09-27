from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import permutation_test_score

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

# 已锁定 sklearn 1.9.1；该兼容接口仍有效，仅屏蔽其未来弃用通知。
warnings.filterwarnings('ignore',message='The `probability` parameter was deprecated',category=FutureWarning)


NON_FEATURE_COLUMNS = {"subject_id", "split", "label", "diagnosis", "qc_ncc"}


def _pipeline(random_state: int) -> tuple[Pipeline, list[dict[str, Any]]]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=50)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    grid: list[dict[str, Any]] = [
        {
            "select__k": [20, 50, 100, "all"],
            "classifier": [
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state,
                )
            ],
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        },
        {
            "select__k": [20, 50, 100, "all"],
            "classifier": [
                SVC(
                    kernel="linear",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                )
            ],
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        },
    ]
    grid.extend([
        {'select__k':[20,50,100,'all'], 'classifier':[LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')]},
        {'select__k':[50,'all'], 'classifier':[RandomForestClassifier(n_estimators=150,max_depth=4,min_samples_leaf=5,class_weight='balanced',n_jobs=1,random_state=random_state)]},
    ])
    return pipeline, grid


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_asd": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity_hc": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probability: np.ndarray,
    random_state: int,
    iterations: int = 2000,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(random_state)
    values: dict[str, list[float]] = {name: [] for name in _metric_dict(y_true, y_pred, probability)}
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        result = _metric_dict(y_true[indices], y_pred[indices], probability[indices])
        for name, value in result.items():
            values[name].append(value)
    return {
        name: [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        for name, samples in values.items()
    }


def train_nested_cv(
    features_csv: Path,
    labels_csv: Path,
    output_dir: Path,
    random_state: int = 20260924,
    outer_splits: int = 5,
    outer_repeats: int = 5,
    inner_splits: int = 4,
    scoring: str = "balanced_accuracy",
    permutation_count: int = 99,
) -> dict[str, object]:
    features = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)
    train = features.query("split == 'train'").merge(labels, on="subject_id", validate="one_to_one")
    if set(train.subject_id) != set(labels.subject_id):
        raise ValueError('训练特征不完整，禁止静默丢弃样本')
    feature_names = [column for column in features.columns if column not in NON_FEATURE_COLUMNS]
    x = train[feature_names]
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('特征包含非有限值，请先检查预处理质量')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    y = (train["label"].to_numpy() == 1).astype(np.int8)
    subjects = train["subject_id"].to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline, grid = _pipeline(random_state)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits,
        n_repeats=outer_repeats,
        random_state=random_state,
    )
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for fold, (train_index, valid_index) in enumerate(outer_cv.split(x, y), start=1):
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + fold)
        search = GridSearchCV(
            clone(pipeline),
            grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=2,
            refit=True,
        )
        search.fit(x.iloc[train_index], y[train_index])
        probability = search.predict_proba(x.iloc[valid_index])[:, 1]
        prediction = search.predict(x.iloc[valid_index])
        print(f'Fold {fold}/{outer_splits * outer_repeats}: {search.best_score_:.3f}', flush=True)
        metrics = _metric_dict(y[valid_index], prediction, probability)
        fold_rows.append(
            {
                "fold": fold,
                **metrics,
                "best_score_inner": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, default=str),
            }
        )
        for subject_id, truth, pred, prob in zip(
            subjects[valid_index], y[valid_index], prediction, probability, strict=True
        ):
            prediction_rows.append(
                {
                    "fold": fold,
                    "repeat": (fold - 1) // outer_splits + 1,
                    "subject_id": subject_id,
                    "truth_asd": int(truth),
                    "prediction_asd": int(pred),
                    "probability_asd": float(prob),
                }
            )

    fold_table = pd.DataFrame(fold_rows)
    prediction_table = pd.DataFrame(prediction_rows)
    # 每名受试者跨重复折的平均概率，作为无重复的 OOF 评价。
    aggregated = (
        prediction_table.groupby(["subject_id", "truth_asd"], as_index=False)[["probability_asd", "prediction_asd"]]
        .mean()
        .sort_values("subject_id")
    )
    aggregated['vote_fraction_asd'] = aggregated['prediction_asd']
    aggregated["prediction_asd"] = (aggregated['vote_fraction_asd'] > 0.5).astype(np.int8)
    y_true = aggregated["truth_asd"].to_numpy()
    y_pred = aggregated["prediction_asd"].to_numpy()
    probability = aggregated["probability_asd"].to_numpy()
    metrics = _metric_dict(y_true, y_pred, probability)
    confidence_intervals = _bootstrap_ci(y_true, y_pred, probability, random_state)
    repeat_metrics = []
    for repeat, block in prediction_table.groupby('repeat'):
        repeat_metrics.append({'repeat':int(repeat), **_metric_dict(block.truth_asd.to_numpy(),block.prediction_asd.to_numpy(),block.probability_asd.to_numpy())})
    pd.DataFrame(repeat_metrics).to_csv(output_dir/'metrics_by_repeat.csv',index=False)

    final_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    final_search = GridSearchCV(
        clone(pipeline), grid, scoring=scoring, cv=final_cv, n_jobs=2, refit=True
    )
    final_search.fit(x, y)
    # 固定基线的完整重训练置换检验，不将其 p 值解释为最终择优模型的 p 值。
    permutation_result = None
    if permutation_count:
        baseline = clone(pipeline).set_params(select__k=20,classifier=LogisticRegression(C=0.1,class_weight='balanced',max_iter=5000,random_state=random_state))
        score,null_scores,p_value=permutation_test_score(baseline,x,y,cv=final_cv,scoring=scoring,n_permutations=permutation_count,n_jobs=2,random_state=random_state)
        permutation_result={'model':'prespecified LogisticRegression C=0.1, SelectKBest k=20','n_permutations':permutation_count,'observed_balanced_accuracy':float(score),'p_value':float(p_value),'scope':'fixed baseline only; not selected final model'}
        pd.DataFrame({'null_balanced_accuracy':null_scores}).to_csv(output_dir/'baseline_permutation_scores.csv',index=False)
    bundle = {
        "estimator": final_search.best_estimator_,
        "feature_names": feature_names,
        "label_mapping": {"ASD": 1, "HC": 2},
        "best_params": final_search.best_params_,
        "random_state": random_state,
    }
    joblib.dump(bundle, output_dir / "final_model.joblib")
    fold_table.to_csv(output_dir / "nested_cv_folds.csv", index=False)
    prediction_table.to_csv(output_dir / "nested_cv_predictions_all_repeats.csv", index=False)
    aggregated.to_csv(output_dir / "nested_cv_predictions_aggregated.csv", index=False)

    fpr, tpr, _ = roc_curve(y_true, probability)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="1 - 特异度", ylabel="灵敏度", title="嵌套交叉验证 ROC")
    axes[0].legend()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[1].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["HC", "ASD"],
        yticklabels=["HC", "ASD"],
        xlabel="预测",
        ylabel="真实",
        title="聚合 OOF 混淆矩阵",
    )
    fig.savefig(output_dir / "model_evaluation.png", dpi=160)
    plt.close(fig)

    summary: dict[str, object] = {
        "n_subjects": int(len(train)),
        "n_features": int(len(feature_names)),
        "outer_splits": outer_splits,
        "outer_repeats": outer_repeats,
        "metrics": metrics,
        "metrics_interpretation": "majority-vote repeated OOF ensemble; single-fit generalization reported separately by repeat",
        "mean_repeat_metrics": pd.DataFrame(repeat_metrics).drop(columns='repeat').mean().to_dict(),
        "ci_interpretation": "descriptive subject bootstrap of fixed OOF predictions; not full refitting uncertainty",
        "baseline_permutation": permutation_result,
        "confidence_intervals_95": confidence_intervals,
        "fold_metric_mean": fold_table.select_dtypes(include="number").mean().to_dict(),
        "fold_metric_std": fold_table.select_dtypes(include="number").std().to_dict(),
        "final_best_params": {key: str(value) for key, value in final_search.best_params_.items()},
        "final_inner_score": float(final_search.best_score_),
    }
    (output_dir / "model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def predict_test(
    features_csv: Path,
    model_path: Path,
    submission_template: Path,
    output_dir: Path,
) -> pd.DataFrame:
    features = pd.read_csv(features_csv)
    test = features.query("split == 'test'").copy()
    bundle = joblib.load(model_path)
    probability = bundle["estimator"].predict_proba(test[bundle["feature_names"]])[:, 1]
    predicted_asd = bundle['estimator'].predict(test[bundle['feature_names']]) == 1
    raw = pd.DataFrame(
        {
            "subject_id": test["subject_id"],
            "probability_asd": probability,
            "label_default": np.where(predicted_asd, 1, 2),
        }
    )
    ranked = raw.sort_values("probability_asd", ascending=False).copy()
    balanced_asd = set(ranked.head(5)["subject_id"])
    raw["label_balanced_5_5"] = np.where(raw["subject_id"].isin(balanced_asd), 1, 2)
    template = pd.read_csv(submission_template)
    if len(test) != 10 or len(template) != 10 or set(test.subject_id) != set(template.subject_id):
        raise ValueError('盲测预测必须与模板完整对应，恰好 10 例')
    submission = template[["subject_id"]].merge(
        raw[["subject_id", "label_balanced_5_5"]], on="subject_id", how="left", validate="one_to_one"
    )
    if submission["label_balanced_5_5"].isna().any():
        raise ValueError("提交模板中存在未预测的 subject_id")
    submission = submission.rename(columns={"label_balanced_5_5": "label"})
    submission["label"] = submission["label"].astype(int)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.sort_values("subject_id").to_csv(output_dir / "test_probabilities.csv", index=False)
    submission.to_csv(output_dir / "submission.csv", index=False)
    default = template[["subject_id"]].merge(
        raw[["subject_id", "label_default"]], on="subject_id", validate="one_to_one"
    ).rename(columns={"label_default": "label"})
    default.to_csv(output_dir / "submission_default_threshold.csv", index=False)
    return submission
