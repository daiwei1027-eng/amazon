from pathlib import Path
import mlflow
import yaml
def log_artifact_file(
    local_path: Path,
    artifact_path: str | None = None,
) -> None:
    if not local_path.is_file():
        raise FileNotFoundError(f"File {local_path} not found")

    if mlflow.active_run() is None:
        raise RuntimeError("No active MLflow run")
    mlflow.log_artifact(
        local_path=str(local_path),
        artifact_path=artifact_path,
    )