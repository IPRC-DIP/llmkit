---
title: "Inference"
subject: Data
license: CC-BY-4.0
keywords: datasets
date: 2024-12-06
authors:
  - name: Ziyuan Nan
    email: nanziyuan21s@ict.ac.cn
    affiliation: ICT CAS
---

## Overview

For inference, we provide a function called `model_map` designed for auto-parallel sampling across GPUs.

### `model_map` Signature

```python3
def model_map(worker, data, model_required_gpus):
```

- **worker**: The inference logic function.
- **data**: The data to be processed.
- **model_required_gpus**: The number of GPUs required per worker.

### How `model_map` Works

1. **GPU Detection**: Reads `CUDA_VISIBLE_DEVICES` from the environment. If not set, it raises an error.
2. **GPU Grouping**: Divides GPUs into `n` groups based on `CUDA_VISIBLE_DEVICES`, `nvidia-smi topo -m`, and `model_required_gpus`.
3. **Data Splitting**: Splits the data into `n` groups.
4. **Worker Execution**: Calls the `worker` function on each group.
5. **Result Sorting**: Sorts and returns the results.

### Worker Function Signature

```python3
def worker(cuda_devices: list[str], data: list[dict[str, Any]])
```

- **cuda_devices**: A list of GPU IDs (e.g., `['1', '2']`).
- **data**: The data for the worker to process.

### Inside the Worker

1. **Set CUDA Visible Devices**: 
   ```python3
   os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_devices)
   ```
2. **Implement Inference Logic**: Write your custom inference logic.

### Important Note

- **Index Preservation**: Each item in `data` is assigned a key `__index__` by `model_map`. To ensure correct sorting of results, do not remove this key.
