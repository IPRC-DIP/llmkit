---
title: "serve"
subject: lmtk
license: CC-BY-4.0
date: 2025-01-09
authors:
  - name: Ziyuan Nan
    email: nanziyuan21s@ict.ac.cn
    affiliation: ICT CAS
---

This project extends the `vllm` serving functionality to support **data parallelism (DP)** for serving large language models (LLMs). It allows you to serve models across multiple GPUs using data parallelism, enabling efficient scaling for high-throughput inference.😊

---

## **Features**
- **Data Parallelism (DP)**: Distribute model inference across multiple GPUs.
- **OpenAI-Compatible API**: Fully compatible with the OpenAI API specification.
- **Multi-GPU Support**: Utilize multiple GPUs by setting `CUDA_VISIBLE_DEVICES`.
- **Easy Integration**: Inherits all parameters from `vllm serve` for seamless compatibility.

---

## **Usage**

### **1. Start the Server**
To start the server, use the following command:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Set visible GPUs
python -m llmkit_data.cli.serve /path/to/model --dtype auto --api-key token-abc123 -dp 4
```

#### **Parameters**
- `/path/to/model`: Path to the model to be served.
- `--dtype auto`: Automatically infer the data type for the model.
- `--api-key token-abc123`: API key for authentication.
- `-dp 4`: Number of data-parallel workers. This must match the total number of GPUs required, calculated as:
  ```
  Total GPUs = Tensor Parallel Size (tp) * Pipeline Parallel Size (pp) * Data Parallel Size (dp)
  ```
  For example, if `tp=1`, `pp=1`, and `dp=4`, then `CUDA_VISIBLE_DEVICES` must include **4 GPUs**.
- Consult the vLLM documentation on the **OpenAI-Compatible Server** for information regarding other parameters: [https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).


#### **Notes**
- **`CUDA_VISIBLE_DEVICES`**: Must be set manually if not using Slurm. This ensures that the server only uses the specified GPUs and avoids conflicts with other users.
- **Slurm**: If running in a Slurm environment, Slurm will automatically set `CUDA_VISIBLE_DEVICES`.

---

### **2. Client Script**
Use the following Python script to interact with the server:

```python
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the OpenAI client
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Server URL
    api_key="token-abc123",               # API key
)

# Function to make a single API call
def generate_completion(messages):
    completion = client.chat.completions.create(
        model="/path/to/model",  # Model path (same as server)
        messages=messages,
    )
    return completion.choices[0].message

# List of messages to process
messages_list = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "user", "content": "Explain the concept of multithreading."}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Tell me a joke."}],
]

# Function to make multithreaded API calls
def multithread_openai_calls(messages_list):
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
        # Submit tasks to the executor
        future_to_messages = {executor.submit(generate_completion, messages): messages for messages in messages_list}

        # Process results as they complete
        for future in as_completed(future_to_messages):
            messages = future_to_messages[future]
            try:
                result = future.result()
                results.append((messages, result))
            except Exception as e:
                print(f"Error processing messages '{messages}': {e}")
                results.append((messages, None))
    return results

# Run the multithreaded OpenAI calls
results = multithread_openai_calls(messages_list)

# Print the results
for messages, result in results:
    print(f"Messages: {messages}")
    print(f"Completion: {result}")
    print("-" * 50)
```

---

## **Design**

### **Data Parallelism (DP)**
The `llmkit_data.cli.serve` module extends `vllm serve` to support **data parallelism**. This allows the model to be served across multiple GPUs, with each GPU handling a portion of the incoming requests. Key design points:
- **Worker Processes**: Each GPU runs a separate worker process, managed by the main server.
- **Load Balancing**: Requests are distributed evenly across workers.
- **Compatibility**: All parameters from `vllm serve` are inherited, ensuring compatibility with existing workflows.

### **GPU Management**

- **Topology-Based Grouping**:
  The server uses `nvidia-smi topo -m` to retrieve the GPU topology matrix and groups GPUs to minimize communication costs within each group.


## **Requirements**

- `vllm` (see [vllm documentation](https://docs.vllm.ai/en/latest/))
