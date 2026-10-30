"""One-to-one temporal matching against independent crossing annotations."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def score_events(labels, predictions, tolerance=0.5):
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    matches = 0
    if labels and predictions:
        costs = np.full((len(labels), len(predictions)), 1e6)
        for i, truth in enumerate(labels):
            for j, pred in enumerate(predictions):
                same = all(truth[k] == pred[k] for k in ("camera", "direction", "class_id"))
                delta = abs(truth["source_seconds"] - pred["source_seconds"])
                if same and delta <= tolerance:
                    costs[i, j] = delta / max(1, tolerance)
        rr, cc = linear_sum_assignment(costs)
        matches = sum(costs[r, c] < 1e6 for r, c in zip(rr, cc))
    precision = matches / len(predictions) if predictions else (1.0 if not labels else 0.0)
    recall = matches / len(labels) if labels else (1.0 if not predictions else 0.0)
    return {
        "kind": "counting_evaluation",
        "labels": len(labels),
        "predictions": len(predictions),
        "true_positive": int(matches),
        "false_positive": len(predictions) - int(matches),
        "false_negative": len(labels) - int(matches),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "absolute_count_error": abs(len(labels) - len(predictions)),
        "tolerance_seconds": tolerance,
    }
