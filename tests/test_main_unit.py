
import argparse

import numpy as np
import pandas as pd
import pytest

from test_deduplicator import main as dedupe_main


def test_env_text_prefers_default_when_missing(monkeypatch):
    monkeypatch.delenv("TEST_DEDUPLICATOR_ID_COL", raising=False)

    assert dedupe_main._env_text("TEST_DEDUPLICATOR_ID_COL", "ID") == "ID"


def test_resolve_text_prefers_cli_over_env(monkeypatch):
    monkeypatch.setenv("TEST_DEDUPLICATOR_ID_COL", "ENV_ID")

    assert dedupe_main._resolve_text("TEST_DEDUPLICATOR_ID_COL", "--id-col", "CLI_ID") == "CLI_ID"


def test_resolve_text_uses_env_when_cli_missing(monkeypatch):
    monkeypatch.setenv("TEST_DEDUPLICATOR_ID_COL", "ENV_ID")

    assert dedupe_main._resolve_text("TEST_DEDUPLICATOR_ID_COL", "--id-col", None) == "ENV_ID"


def test_resolve_text_raises_when_missing(monkeypatch):
    monkeypatch.delenv("TEST_DEDUPLICATOR_ID_COL", raising=False)

    with pytest.raises(ValueError, match="--id-col"):
        dedupe_main._resolve_text("TEST_DEDUPLICATOR_ID_COL", "--id-col", None)


def test_parse_arguments_uses_environment_defaults_except_file(monkeypatch):
    monkeypatch.setenv("TEST_DEDUPLICATOR_FILE", "sample_data/custom.csv")
    monkeypatch.setenv("TEST_DEDUPLICATOR_THRESHOLD", "0.91")
    monkeypatch.setenv("TEST_DEDUPLICATOR_TITLE_WEIGHT", "0.7")
    monkeypatch.setenv("TEST_DEDUPLICATOR_BODY_WEIGHT", "0.3")
    monkeypatch.setenv("OPENAI_API_BASE", "http://example.test/v1")
    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "--id-col", "ID", "--title-col", "Title", "--body-col", "Steps", "--model", "model-x"],
    )

    args = dedupe_main.parse_arguments()

    assert args.file is None
    assert args.threshold == pytest.approx(0.91)
    assert args.title_weight == pytest.approx(0.7)
    assert args.body_weight == pytest.approx(0.3)
    assert args.model == "model-x"


def test_resolve_required_settings_prefers_cli_values(monkeypatch):
    monkeypatch.setenv("TEST_DEDUPLICATOR_FILE", "sample_data/from-env.csv")
    monkeypatch.setenv("TEST_DEDUPLICATOR_ID_COL", "ENV_ID")
    monkeypatch.setenv("TEST_DEDUPLICATOR_TITLE_COL", "ENV_TITLE")
    monkeypatch.setenv("TEST_DEDUPLICATOR_BODY_COL", "ENV_BODY")
    monkeypatch.setenv("TEST_DEDUPLICATOR_MODEL", "env-model")

    args = argparse.Namespace(
        file="cli.csv",
        id_col="CLI_ID",
        title_col="CLI_TITLE",
        body_col="CLI_BODY",
        model="cli-model",
        threshold=0.85,
        title_weight=0.6,
        body_weight=0.4,
    )

    resolved = dedupe_main.resolve_required_settings(args)

    assert resolved.file == "cli.csv"
    assert resolved.id_col == "CLI_ID"
    assert resolved.title_col == "CLI_TITLE"
    assert resolved.body_col == "CLI_BODY"
    assert resolved.model == "cli-model"


def test_resolve_required_settings_uses_env_when_cli_missing(monkeypatch):
    monkeypatch.setenv("TEST_DEDUPLICATOR_FILE", "sample_data/from-env.csv")
    monkeypatch.setenv("TEST_DEDUPLICATOR_ID_COL", "ENV_ID")
    monkeypatch.setenv("TEST_DEDUPLICATOR_TITLE_COL", "ENV_TITLE")
    monkeypatch.setenv("TEST_DEDUPLICATOR_BODY_COL", "ENV_BODY")
    monkeypatch.setenv("TEST_DEDUPLICATOR_MODEL", "env-model")

    args = argparse.Namespace(
        file=None,
        id_col=None,
        title_col=None,
        body_col=None,
        model=None,
        threshold=0.85,
        title_weight=0.6,
        body_weight=0.4,
    )

    resolved = dedupe_main.resolve_required_settings(args)

    assert resolved.file == "sample_data/from-env.csv"
    assert resolved.id_col == "ENV_ID"
    assert resolved.title_col == "ENV_TITLE"
    assert resolved.body_col == "ENV_BODY"
    assert resolved.model == "env-model"


def test_resolve_required_settings_raises_when_missing_required_values(monkeypatch):
    monkeypatch.delenv("TEST_DEDUPLICATOR_FILE", raising=False)
    monkeypatch.delenv("TEST_DEDUPLICATOR_ID_COL", raising=False)
    monkeypatch.delenv("TEST_DEDUPLICATOR_TITLE_COL", raising=False)
    monkeypatch.delenv("TEST_DEDUPLICATOR_BODY_COL", raising=False)
    monkeypatch.delenv("TEST_DEDUPLICATOR_MODEL", raising=False)

    args = argparse.Namespace(
        file=None,
        id_col=None,
        title_col=None,
        body_col=None,
        model=None,
        threshold=0.85,
        title_weight=0.6,
        body_weight=0.4,
    )

    with pytest.raises(ValueError, match="--file"):
        dedupe_main.resolve_required_settings(args)


def test_validate_required_columns_reports_missing_columns(capsys):
    df = pd.DataFrame({"ID": [1], "Title": ["A"]})

    assert dedupe_main.validate_required_columns(df, ["ID", "Title", "Steps"]) is False

    output = capsys.readouterr().out
    assert "Missing expected columns" in output


def test_prepare_test_cases_filters_and_fills_missing_values(capsys):
    df = pd.DataFrame(
        {
            "ID": ["T1", None, "T3"],
            "Title": ["Login", "Checkout", None],
            "Steps": [None, "Do checkout", "Missing title"],
        }
    )

    cleaned = dedupe_main.prepare_test_cases(df, "ID", "Title", "Steps")

    assert cleaned["ID"].tolist() == ["T1"]
    assert cleaned["Steps"].tolist() == [""]
    assert "Loaded 1 valid test cases." in capsys.readouterr().out


def test_extract_semantic_text_fields_returns_lists():
    df = pd.DataFrame({"Title": ["Login", "Checkout"], "Steps": ["Open app", "Pay"]})

    titles, bodies = dedupe_main.extract_semantic_text_fields(df, "Title", "Steps")

    assert titles == ["Login", "Checkout"]
    assert bodies == ["Open app", "Pay"]


def test_build_weighted_similarity_matrix_uses_title_and_body_weights(monkeypatch):
    class FakeEncoder:
        def embed_documents(self, documents):
            return [[1.0, 0.0] if text == "A" else [0.0, 1.0] for text in documents]

    monkeypatch.setattr(dedupe_main, "get_embedding_client", lambda model_name: FakeEncoder())

    similarity_matrix = dedupe_main.build_weighted_similarity_matrix(
        model_name="model-x",
        titles=["A", "B"],
        bodies=["B", "A"],
        title_weight=0.6,
        body_weight=0.4,
    )

    assert similarity_matrix is not None
    assert similarity_matrix.shape == (2, 2)
    assert similarity_matrix[0][1] == pytest.approx(similarity_matrix[1][0])


def test_report_duplicate_clusters_prints_expected_match(capsys):
    df = pd.DataFrame({"ID": ["T1", "T2"], "Title": ["Login", "Log in"]})
    similarity_matrix = np.array([[1.0, 0.9], [0.9, 1.0]])

    dedupe_main.report_duplicate_clusters(df, similarity_matrix, "ID", "Title", 0.8)

    output = capsys.readouterr().out
    assert "Match Found" in output
    assert "T1" in output and "T2" in output
    assert "Found 1 suspect duplicate clusters" in output