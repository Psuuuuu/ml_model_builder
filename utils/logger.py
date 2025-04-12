import os
import json
import time
from datetime import datetime
import joblib
import psutil
import platform

try:
    import cpuinfo
    CPU_NAME = cpuinfo.get_cpu_info().get('brand_raw', platform.processor())
except Exception:
    CPU_NAME = platform.processor()

def log_experiment_results(logs, log_dir="experiments/logs/", log_file="experiment_log.jsonl"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    with open(log_path, "a") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")

def create_log_entry(experiment_title, model_name, hyperparams, dataset_name, preprocessing, metrics, train_time, model_object):
    timestamp = datetime.now().isoformat()
    model_size = get_model_size(model_object)
    cpu_util = psutil.cpu_percent(interval=0.1)
    return {
        "experiment_title": experiment_title,
        "timestamp": timestamp,
        "model": model_name,
        "hyperparameters": hyperparams,
        "dataset": dataset_name,
        "preprocessing": preprocessing,
        "metrics": metrics,
        "training_time_sec": train_time,
        "model_size_bytes": model_size,
        "system_info": {
            "cpu": CPU_NAME,
            "cpu_utilization": cpu_util,
            "memory_used_mb": psutil.Process().memory_info().rss // 1024 ** 2
        }
    }

def get_model_size(model):
    temp_path = "experiments/logs/_temp_model.joblib"
    joblib.dump(model, temp_path)
    size = os.path.getsize(temp_path)
    os.remove(temp_path)
    return size
