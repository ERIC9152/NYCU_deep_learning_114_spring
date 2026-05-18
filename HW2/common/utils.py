
import os, random, numpy as np, tensorflow as tf

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def param_count(model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))

def save_history(history, out_json):
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(out_json, "w", encoding="utf-8") as f:
        import json; json.dump(hist, f, indent=2)

def time_stamp():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
