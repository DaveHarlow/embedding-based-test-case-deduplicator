import json
import subprocess
import time

import pytest
import requests

# Model identifier exposed by the local OpenAI-compatible API.
TARGET_MODEL_ID = "text-embedding-nomic-embed-text-v1.5@f32"
# Model key shown by `lms ls --embedding --json`.
TARGET_MODEL_KEY = "nomic-ai/nomic-embed-text-v1.5-GGUF@f32"
# Download source accepted by `lms get`.
TARGET_MODEL_SOURCE = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF@f32"
LOCAL_URL = "http://localhost:1234/v1/models"
SERVER_WAIT_SECONDS = 30
MODEL_WAIT_SECONDS = 90
MODEL_DOWNLOAD_WAIT_SECONDS = 1800
POLL_INTERVAL_SECONDS = 2


def _get_loaded_models() -> list[str]:
    response = requests.get(LOCAL_URL, timeout=3)
    response.raise_for_status()
    return [model["id"] for model in response.json().get("data", [])]


def _wait_for_server(timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            requests.get(LOCAL_URL, timeout=2).raise_for_status()
            return True
        except requests.exceptions.RequestException:
            time.sleep(POLL_INTERVAL_SECONDS)
    return False


def _wait_for_model(model_id: str, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if model_id in _get_loaded_models():
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(POLL_INTERVAL_SECONDS)
    return False


def _model_available_on_disk(model_key: str) -> bool:
    try:
        result = subprocess.run(
            ["lms", "ls", "--embedding", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    try:
        models = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False

    return any(model.get("modelKey") == model_key for model in models)


@pytest.fixture(scope="session")
def ensure_lm_studio_infrastructure():
    """
    Session-wide fixture that checks if LM Studio is up and hosting the correct model.
    If not, it boots the daemon, starts the server, and loads the model automatically.
    """
    print("\n[Infra Setup] Checking local LM Studio status...")

    # 1. Check if the server endpoint is active
    server_running = _wait_for_server(timeout_seconds=2)

    # 2. If server is down, spin up the daemon and start the server
    if not server_running:
        print("[Infra Setup] Server offline. Booting LM Studio daemon and server via CLI...")
        try:
            # Ensure the system-level daemon is active
            subprocess.run(["lms", "daemon", "up"], check=True, capture_output=True)
            # Start the local HTTP server in a non-blocking process and then poll readiness.
            subprocess.Popen(["lms", "server", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not _wait_for_server(timeout_seconds=SERVER_WAIT_SECONDS):
                pytest.exit("❌ LM Studio server did not become ready in time.")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.exit("❌ Critical Error: 'lms' CLI tool not found in PATH or failed to start. Is LM Studio installed?")

    # 3. Verify if our specific embedding model is loaded into memory
    print(f"[Infra Setup] Ensuring model '{TARGET_MODEL_ID}' is loaded...")
    try:
        loaded_models = _get_loaded_models()

        if TARGET_MODEL_ID not in loaded_models:
            if not _model_available_on_disk(TARGET_MODEL_KEY):
                print(
                    f"[Infra Setup] Model '{TARGET_MODEL_ID}' is not on disk. "
                    "Downloading it now (first run may download ~500MB)..."
                )
                subprocess.run(
                    ["lms", "get", TARGET_MODEL_SOURCE, "-y"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=MODEL_DOWNLOAD_WAIT_SECONDS,
                )

            print(f"[Infra Setup] Model not in memory. Loading '{TARGET_MODEL_ID}' dynamically...")
            # Trigger model load and wait until the model appears in /models.
            subprocess.run(
                ["lms", "load", TARGET_MODEL_ID],
                check=True,
                capture_output=True,
                text=True,
                timeout=MODEL_WAIT_SECONDS,
            )
            if not _wait_for_model(TARGET_MODEL_ID, timeout_seconds=MODEL_WAIT_SECONDS):
                pytest.exit(
                    f"❌ Timed out waiting for model '{TARGET_MODEL_ID}' to load. "
                    "Model loading from UI can take ~20s; increase MODEL_WAIT_SECONDS if needed."
                )
            print("✅ Model loaded successfully.")
        else:
            print(f"✅ Model '{TARGET_MODEL_ID}' is already loaded and ready.")

    except subprocess.CalledProcessError as e:
        stdout = (e.stdout or "").strip() if hasattr(e, "stdout") else ""
        stderr = (e.stderr or "").strip() if hasattr(e, "stderr") else ""
        debug_output = "\n".join(part for part in [f"stdout: {stdout}" if stdout else "", f"stderr: {stderr}" if stderr else ""] if part)
        pytest.exit(
            f"❌ LM Studio CLI command failed while preparing model '{TARGET_MODEL_ID}': {e}"
            + (f"\n{debug_output}" if debug_output else "")
        )
    except Exception as e:
        pytest.exit(f"❌ Failed to verify or load model over HTTP endpoint: {e}")

    yield

    # 4. Best-effort teardown strategy depends on initial server state.
    # If LM Studio was already running, only unload the model we used.
    # If we had to start LM Studio, stop server/daemon to avoid leaving services running.
    if server_running:
        print(f"[Infra Teardown] Attempting to unload model '{TARGET_MODEL_ID}'...")
        try:
            subprocess.run(
                ["lms", "unload", TARGET_MODEL_ID],
                check=True,
                capture_output=True,
                text=True,
                timeout=MODEL_WAIT_SECONDS,
            )
            if _wait_for_model(TARGET_MODEL_ID, timeout_seconds=2):
                print(
                    f"[Infra Teardown] Model '{TARGET_MODEL_ID}' still appears loaded after unload request. "
                    "Continuing without failing tests."
                )
            else:
                print(f"[Infra Teardown] Model '{TARGET_MODEL_ID}' unloaded successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"[Infra Teardown] Warning: failed to unload model '{TARGET_MODEL_ID}': {exc}")
    else:
        print("[Infra Teardown] Fixture started LM Studio. Attempting to stop server and daemon...")
        try:
            subprocess.run(
                ["lms", "server", "stop"],
                check=True,
                capture_output=True,
                text=True,
                timeout=MODEL_WAIT_SECONDS,
            )
            print("[Infra Teardown] LM Studio server stopped.")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"[Infra Teardown] Warning: failed to stop LM Studio server: {exc}")

        try:
            subprocess.run(
                ["lms", "daemon", "down"],
                check=True,
                capture_output=True,
                text=True,
                timeout=MODEL_WAIT_SECONDS,
            )
            print("[Infra Teardown] LM Studio daemon stopped.")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
            print(f"[Infra Teardown] Warning: failed to stop LM Studio daemon: {exc}")

