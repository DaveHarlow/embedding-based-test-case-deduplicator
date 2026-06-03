import re
import sys
from pathlib import Path

from test_deduplicator import main as dedupe_main


def test_main_workflow_detects_duplicate_pair_from_csv(capsys, ensure_lm_studio_infrastructure):
    """Runs the full main.py workflow from CSV input and validates output summary."""
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "sample_data" / "sample_cases.csv"
    assert csv_path.exists(), f"Missing sample CSV file: {csv_path}"

    argv = [
        "main.py",
        "--file",
        str(csv_path),
        "--id-col",
        "ID",
        "--title-col",
        "Title",
        "--body-col",
        "Steps",
        "--threshold",
        "0.75",
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        dedupe_main.main()
    finally:
        sys.argv = old_argv

    output = capsys.readouterr().out

    assert "✅ Loaded 3 valid test cases." in output
    assert "Analysis complete." in output
    assert "❌" not in output

    match = re.search(r"Found\s+(\d+)\s+suspect duplicate clusters", output)
    assert match is not None, f"Could not find duplicate summary in output:\n{output}"
    assert int(match.group(1)) >= 1, f"Expected at least one duplicate cluster. Output:\n{output}"

    # Ensure the expected near-duplicate IDs were surfaced as a match.
    assert "T1" in output and "T2" in output