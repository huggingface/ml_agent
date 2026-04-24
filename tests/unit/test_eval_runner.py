import pytest

from agent.eval.registry import EvalTask
from agent.eval.registry import get_task
from agent.eval.runner import evaluate_model, load_examples, normalize_label


class FakeInferenceClient:
    def __init__(self, predictions):
        self._predictions = predictions
        self.calls = []

    def text_classification(self, text, model):
        self.calls.append((text, model))
        label = self._predictions[len(self.calls) - 1]
        return [{"label": label, "score": 0.99}]


class FakeLabelObject:
    def __init__(self, label):
        self.label = label


class FakeObjectInferenceClient(FakeInferenceClient):
    def text_classification(self, text, model):
        self.calls.append((text, model))
        label = self._predictions[len(self.calls) - 1]
        return FakeLabelObject(label)


def test_normalize_label_supports_hf_text_classification_aliases():
    assert normalize_label("NEGATIVE") == 0
    assert normalize_label("POSITIVE") == 1
    assert normalize_label("LABEL_0") == 0
    assert normalize_label("LABEL_1") == 1
    assert normalize_label("0") == 0
    assert normalize_label("1") == 1
    assert normalize_label(0) == 0
    assert normalize_label(1) == 1


def test_evaluate_model_computes_accuracy_from_examples():
    task = get_task("glue_sst2")
    examples = [
        {"sentence": "bad film", "label": 0},
        {"sentence": "great film", "label": 1},
        {"sentence": "fine film", "label": 1},
    ]
    client = FakeInferenceClient(["NEGATIVE", "POSITIVE", "NEGATIVE"])

    result = evaluate_model(
        task=task,
        model_id="candidate-model",
        examples=examples,
        client=client,
    )

    assert result.model_id == "candidate-model"
    assert result.metrics == {"accuracy": 2 / 3}
    assert len(client.calls) == 3


def test_evaluate_model_accepts_object_label_responses():
    task = get_task("glue_sst2")
    examples = [{"sentence": "great film", "label": 1}]
    client = FakeObjectInferenceClient(["POSITIVE"])

    result = evaluate_model(
        task=task,
        model_id="candidate-model",
        examples=examples,
        client=client,
    )

    assert result.metrics == {"accuracy": 1.0}


def test_load_examples_respects_limit(monkeypatch):
    class FakeSplit(list):
        pass

    def fake_load_dataset(name, config, split):
        assert name == "glue"
        assert config == "sst2"
        assert split == "validation"
        return FakeSplit(
            [
                {"sentence": "a", "label": 0},
                {"sentence": "b", "label": 1},
                {"sentence": "c", "label": 0},
            ]
        )

    monkeypatch.setattr("agent.eval.runner.load_dataset", fake_load_dataset)

    task = get_task("glue_sst2")
    examples = load_examples(task, split="validation", limit=2)

    assert examples == [
        {"sentence": "a", "label": 0},
        {"sentence": "b", "label": 1},
    ]


def test_load_examples_omits_missing_dataset_config(monkeypatch):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"text": "a", "label": 0}]

    monkeypatch.setattr("agent.eval.runner.load_dataset", fake_load_dataset)

    task = EvalTask(
        task_id="custom",
        dataset_name="custom_dataset",
        dataset_config=None,
        default_split="test",
        text_column="text",
        label_column="label",
        primary_metric="accuracy",
    )

    assert load_examples(task) == [{"text": "a", "label": 0}]
    assert calls == [(("custom_dataset",), {"split": "test"})]


def test_evaluate_model_reports_failed_example_index(capsys):
    class FailingInferenceClient:
        def __init__(self):
            self.calls = 0

        def text_classification(self, text, model):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("rate limited")
            return [{"label": "NEGATIVE", "score": 0.99}]

    task = get_task("glue_sst2")
    examples = [
        {"sentence": "bad film", "label": 0},
        {"sentence": "great film", "label": 1},
    ]

    with pytest.raises(RuntimeError, match="rate limited"):
        evaluate_model(
            task=task,
            model_id="candidate-model",
            examples=examples,
            client=FailingInferenceClient(),
        )

    captured = capsys.readouterr()
    assert "task=glue_sst2" in captured.err
    assert "model=candidate-model" in captured.err
    assert "example_index=1" in captured.err
