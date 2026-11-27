"""DDI 事件预测指标和逐药对输出。"""

import csv

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(labels, predictions):
    # 沿用历史口径，宏平均覆盖真实值或预测值出现过的类别。
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }


def write_predictions(path, records, predictions):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["smile_1", "smile_2", "label", "prediction"])
        for pair, prediction in zip(records, predictions, strict=True):
            writer.writerow([*pair, prediction])
