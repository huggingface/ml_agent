"""GLUE SST-2 dataset loading and evaluation helpers."""

import sys

from datasets import load_dataset
from huggingface_hub import InferenceClient

from agent.eval.compare import ModelResult
from agent.eval.registry import EvalTask


def normalize_label(label: str | int) -> int:
    mapping = {
        "NEGATIVE": 0,
        "POSITIVE": 1,
        "LABEL_0": 0,
        "LABEL_1": 1,
        "0": 0,
        "1": 1,
    }
    if isinstance(label, int):
        if label in (0, 1):
            return label
        raise ValueError(f"Unsupported label from inference API: {label}")
    try:
        return mapping[str(label).upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported label from inference API: {label}") from exc


def load_examples(
    task: EvalTask,
    split: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    selected_split = split or task.default_split
    dataset_args = [task.dataset_config] if task.dataset_config else []
    dataset = load_dataset(task.dataset_name, *dataset_args, split=selected_split)
    records = list(dataset)
    if limit is not None:
        records = records[:limit]
    return records


def extract_label(response) -> str | int:
    if isinstance(response, list):
        if not response:
            raise ValueError("Empty response from inference API")
        response = response[0]

    if isinstance(response, dict):
        return response["label"]

    label = getattr(response, "label", None)
    if label is None:
        raise ValueError("Inference response does not contain a label")
    return label


def evaluate_model(
    task: EvalTask,
    model_id: str,
    examples: list[dict],
    client: InferenceClient | None = None,
) -> ModelResult:
    client = client or InferenceClient()
    correct = 0

    for index, example in enumerate(examples):
        try:
            response = client.text_classification(example[task.text_column], model=model_id)
            predicted = normalize_label(extract_label(response))
        except Exception as exc:
            print(
                "Evaluation failed "
                f"(task={task.task_id}, model={model_id}, example_index={index}): {exc}",
                file=sys.stderr,
            )
            raise
        if predicted == example[task.label_column]:
            correct += 1

    accuracy = correct / len(examples) if examples else 0.0
    return ModelResult(model_id=model_id, metrics={"accuracy": accuracy})
