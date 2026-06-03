import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import SecretStr
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Load local LM Studio configuration from the repository-level .env file.
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_THRESHOLD = 0.85
DEFAULT_TITLE_WEIGHT = 0.60
DEFAULT_BODY_WEIGHT = 0.40
DEFAULT_OPENAI_API_BASE = "http://localhost:1234/v1"


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _resolve_text(name: str, cli_flag: str, cli_value: str | None) -> str:
    if cli_value is not None and cli_value != "":
        return cli_value

    env_value = os.getenv(name)
    if env_value is not None and env_value != "":
        return env_value

    raise ValueError(f"Missing required setting: provide {cli_flag} or set {name} in .env")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default

    try:
        return float(value)
    except ValueError:
        return default


def _resolve_input_path(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.exists():
        return candidate

    repo_relative_candidate = PROJECT_ROOT / candidate
    if repo_relative_candidate.exists():
        return repo_relative_candidate

    return candidate


def get_embedding_client(model_name: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        # For local LM Studio, the base URL typically points to localhost with the appropriate port
        base_url=_env_text("OPENAI_API_BASE", DEFAULT_OPENAI_API_BASE),
        # For local LM Studio, the API key can be a dummy value since auth is typically disabled
        api_key=SecretStr(_env_text("OPENAI_API_KEY", "dummy-key-for-local-lm-studio")),
        # the model we downloaded earlier
        model=model_name,
        # We want to embed the combined title+body context as a single unit, so we disable chunking and context length checks
        chunk_size=8191,
        check_embedding_ctx_length=False
    )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AI-Powered QA Test Case Deduplicator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", default=None, help="Path to the test case CSV import file")
    parser.add_argument("--id-col", default=None, help="Name of the unique ID column (e.g., 'Issue Key', 'ID')")
    parser.add_argument("--title-col", default=None, help="Name of the test title/summary column")
    parser.add_argument("--body-col", default=None, help="Name of the steps/description/pre-conditions column")
    parser.add_argument("--model", default=None, help="Embedding model name used by the local inference endpoint")
    parser.add_argument("--threshold", type=float, default=_env_float("TEST_DEDUPLICATOR_THRESHOLD", 0.85), help="Similarity threshold between 0.0 and 1.0")
    parser.add_argument("--title-weight", type=float, default=_env_float("TEST_DEDUPLICATOR_TITLE_WEIGHT", 0.60), help="Weight applied to the title embeddings")
    parser.add_argument("--body-weight", type=float, default=_env_float("TEST_DEDUPLICATOR_BODY_WEIGHT", 0.40), help="Weight applied to the body embeddings")
    return parser.parse_args()


def resolve_required_settings(args: argparse.Namespace) -> argparse.Namespace:
    args.file = _resolve_text("TEST_DEDUPLICATOR_FILE", "--file", args.file)
    args.id_col = _resolve_text("TEST_DEDUPLICATOR_ID_COL", "--id-col", args.id_col)
    args.title_col = _resolve_text("TEST_DEDUPLICATOR_TITLE_COL", "--title-col", args.title_col)
    args.body_col = _resolve_text("TEST_DEDUPLICATOR_BODY_COL", "--body-col", args.body_col)
    args.model = _resolve_text("TEST_DEDUPLICATOR_MODEL", "--model", args.model)
    return args


def load_test_cases(csv_path: Path) -> pd.DataFrame | None:
    print(f"📊 Loading {csv_path}...")
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        print(f"❌ Error reading CSV file: {exc}")
        return None


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> bool:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing expected columns in CSV: {missing}")
        print(f"Available columns are: {list(df.columns)}")
        return False

    return True


def prepare_test_cases(df: pd.DataFrame, id_col: str, title_col: str, body_col: str) -> pd.DataFrame:
    cleaned_df = df.dropna(subset=[id_col, title_col]).copy()
    cleaned_df[body_col] = cleaned_df[body_col].fillna("")
    print(f"✅ Loaded {len(cleaned_df)} valid test cases.")
    return cleaned_df


def extract_semantic_text_fields(df: pd.DataFrame, title_col: str, body_col: str) -> tuple[list[str], list[str]]:
    print("🧠 Isolating semantic text fields...")
    titles = df[title_col].astype(str).tolist()
    bodies = df[body_col].astype(str).tolist()
    return titles, bodies


def build_weighted_similarity_matrix(
    model_name: str,
    titles: list[str],
    bodies: list[str],
    title_weight: float,
    body_weight: float,
) -> np.ndarray | None:
    print("🤖 Batching embeddings via local inference engine...")
    encoder = get_embedding_client(model_name)

    try:
        title_vectors = np.array(encoder.embed_documents(titles))
        body_vectors = np.array(encoder.embed_documents(bodies))
    except Exception as exc:
        print(f"❌ Embedding failed: {exc}")
        return None

    print("📐 Engineering weighted semantic matrix...")
    matrix = (title_vectors * title_weight) + (body_vectors * body_weight)
    return cosine_similarity(matrix)


def report_duplicate_clusters(
    df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    id_col: str,
    title_col: str,
    threshold: float,
) -> None:
    print(f"🔍 Analyzing pairs exceeding similarity threshold >= {threshold}...")
    duplicates_found = 0
    seen_pairs: set[tuple[str, str]] = set()

    print("\n================ DETECTED POTENTIAL DUPLICATES ================")

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            similarity = similarity_matrix[i][j]

            if similarity < threshold:
                continue

            id_a = df.iloc[i][id_col]
            title_a = df.iloc[i][title_col]
            id_b = df.iloc[j][id_col]
            title_b = df.iloc[j][title_col]

            left_id = str(id_a)
            right_id = str(id_b)
            pair_key = (left_id, right_id) if left_id <= right_id else (right_id, left_id)
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            duplicates_found += 1

            print(f"⚠️ Match Found! Match Confidence: {similarity * 100:.2f}%")
            print(f"  [1] {id_a}: {title_a}")
            print(f"  [2] {id_b}: {title_b}")
            print("-" * 63)

    print(f"\nAnalysis complete. Found {duplicates_found} suspect duplicate clusters.")

def main():
    try:
        args = resolve_required_settings(parse_arguments())
    except ValueError as exc:
        print(f"❌ {exc}")
        return

    csv_path = _resolve_input_path(args.file)

    df = load_test_cases(csv_path)
    if df is None:
        return

    if not validate_required_columns(df, [args.id_col, args.title_col, args.body_col]):
        return

    df = prepare_test_cases(df, args.id_col, args.title_col, args.body_col)
    titles, bodies = extract_semantic_text_fields(df, args.title_col, args.body_col)

    similarity_matrix = build_weighted_similarity_matrix(
        model_name=args.model,
        titles=titles,
        bodies=bodies,
        title_weight=args.title_weight,
        body_weight=args.body_weight,
    )
    if similarity_matrix is None:
        return

    report_duplicate_clusters(df, similarity_matrix, args.id_col, args.title_col, args.threshold)

if __name__ == "__main__":
    main()