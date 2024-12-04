---
title: "Contribution"
subject: Data
license: CC-BY-4.0
---

## Project Structure

```bash
src
  └─llmkit_data
    ├─cli (entry point for scripts)
    ├─converter (converts standard datasets to framework-specific formats)
    ├─std (tools for preprocessing standard datasets)
    └─utils
```

**llmkit** is designed to be used both as a library and as a command-line tool.

- **Entry Points**: Only Python files under `cli` can have an entry point like:
  ```python
  if __name__ == "__main__":
      pass
  ```

- **Function Modules**: Other Python modules should be collections of functions. This allows easy importation in other projects or scripts, such as:
  ```python
  from llmkit_data.utils.json import read_jsonl, write_jsonl
  ```

- **Function Scope**: Functions defined in `cli` should only be used within the script and should not be imported by others. If a function might be imported in the future, place it under `cli` temporarily and refactor it later.

## Adding a New Dataset

To add a new dataset, create an entry named `prep_{dataset_name}.py` under `cli`. Users can then use it via:
```bash
python -m llmkit_data.cli.prep_{dataset_name}
```

## Adding a New Converter (Support for a New Framework)

1. **Research**: Read the new framework's documentation to understand its data formats and how it works.
2. **Converter Implementation**:
   - Add a new converter in `converter/{framework}.py`. Most functions should be placed here for easy importation.
   - Create an entry point in `cli/convert_to_{framework}.py`.

**Note**: This command-line tool typically supports various dataset formats. Refer to [Dataset Formats](./format.md) for more details.