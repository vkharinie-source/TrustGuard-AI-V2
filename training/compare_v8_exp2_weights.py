import numpy as np
import tensorflow as tf
from pathlib import Path

base = Path("models/deepfake_image_v8")

paths = {
    "best": base / "deepfake_mobilenetv8_exp2_weighted_best.keras",
    "latest": base / "deepfake_mobilenetv8_exp2_weighted_latest.keras",
}

models = {
    name: tf.keras.models.load_model(str(path), compile=False)
    for name, path in paths.items()
}

best_weights = models["best"].get_weights()
latest_weights = models["latest"].get_weights()

print("Best weight arrays:", len(best_weights))
print("Latest weight arrays:", len(latest_weights))

if len(best_weights) != len(latest_weights):
    print("RESULT: Different numbers of weight arrays.")
else:
    changed = 0
    max_diff = 0.0

    for a, b in zip(best_weights, latest_weights):
        if a.shape != b.shape:
            changed += 1
            print("Shape mismatch:", a.shape, b.shape)
            continue

        diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        max_diff = max(max_diff, diff)

        if not np.array_equal(a, b):
            changed += 1

    print("Weight arrays that differ:", changed)
    print("Maximum absolute weight difference:", max_diff)
    print("Weights identical:", changed == 0)

del models
