---
title: "Datasets format"
subject: Data
license: CC-BY-4.0
keywords: datasets
---

## Standard Dataset format

The dataset should be stored in a JSONL (JSON Lines) file format, where each line represents a separate JSON object.

### Reading and Writing JSONL Files

You can utilize the jsonlines Python package to read or write JSONL files. Alternatively, you can use the following code snippets for basic operations:

```python3
import json

def load_jsonl(file_path):
    """Load a JSONL file into a list of dictionaries."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
```

### Command Line Tools for JSONL Files

Several command line tools can be used to manipulate JSONL files. Personally, I find sed and jq particularly useful.

For instance, to extract the messages from the fifth line of a JSONL file, you can use the following command:

```sh
sed -n '5p' dataset.jsonl | jq .messages
```

## SFT datasets

SFT datasets must adhere to the following format.

```json
{
  "question": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you! How can I assist you today?"
    }
    {
      "role": "user",
      "content": "Can you help me with a JSON schema?"
    },
  ],
  "response": [
    {
      "role": "assistant",
      "content": "Of course! I'd be happy to help you with that."
    }
  ]
}
```

This item should be preprocessed using the following code.

```python3
prompt = apply_chat_template(data["question"], tokenize=False, add_generation_prompt=True)
response = apply_chat_template(data["question"] + data["response"], tokenize=False)[len(prompt) :]
```

## Reward Datasets

Reward datasets must adhere to the following format.

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you! How can I assist you today?"
    },
    {
      "role": "user",
      "content": "Can you help me with a JSON schema?"
    }
  ],
  "chosen": [
    {
      "role": "assistant",
      "content": "Of course! I'd be happy to help you with that."
    }
  ],
  "rejected": [
    {
      "role": "assistant",
      "content": "Sorry, I can't assist with that."
    }
  ]
}
```

This item should be preprocessed using the following code.

```python3
prompt = apply_chat_template(data["prompt"], tokenize=False, add_generation_prompt=True)
chosen = apply_chat_template(data["prompt"] + data["chosen"], tokenize=False)[len(prompt) :]
rejected = apply_chat_template(data["prompt"] + data["rejected"], tokenize=False)[len(prompt) :]
```