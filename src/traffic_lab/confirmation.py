"""Trial-level comparisons and portable integrity checks for the GPU confirmation study."""

import hashlib
import json
import math
import random
from statistics import mean


def json_digest(value):
    """Hash JSON values without depending on checkout line endings."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def balanced_orders(pairs, seed):
    if pairs < 4 or pairs % 2:
        raise ValueError("Use an even number of at least four pairs")
    orders = [["pytorch", "tensorrt"], ["tensorrt", "pytorch"]] * (pairs // 2)
    random.Random(seed).shuffle(orders)
    return orders


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered or not 0 <= fraction <= 1:
        raise ValueError("Nonempty values and a fraction in [0, 1] are required")
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_speedup(trials, seed=20261030, resamples=10000):
    pairs = {}
    for trial in trials:
        if (
            trial["runtime"] not in ("pytorch", "tensorrt")
            or not math.isfinite(trial["fps"])
            or trial["fps"] <= 0
        ):
            raise ValueError("Trials require a known runtime and finite positive FPS")
        group = pairs.setdefault(trial["pair"], {})
        if trial["runtime"] in group:
            raise ValueError("Duplicate runtime in a pair")
        group[trial["runtime"]] = trial["fps"]
    if len(pairs) < 4 or any(set(pair) != {"pytorch", "tensorrt"} for pair in pairs.values()):
        raise ValueError("At least four complete pairs are required")
    logs = [math.log(pair["tensorrt"] / pair["pytorch"]) for _, pair in sorted(pairs.items())]
    rng = random.Random(seed)
    bootstrap = [math.exp(mean(rng.choices(logs, k=len(logs)))) for _ in range(resamples)]
    return {
        "pairs": len(pairs),
        "paired_fps_ratios": [math.exp(value) for value in logs],
        "geometric_mean_speedup": math.exp(mean(logs)),
        "trial_bootstrap_95_interval": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "note": "Exploratory paired-trial interval on one shared host; frames are not independent replicates.",
    }
