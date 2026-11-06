"""真实 MLflow 集成测试：设置 RUN_MLFLOW_TESTS=1 后运行。"""

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MLFLOW_TESTS") != "1",
    reason="设置 RUN_MLFLOW_TESTS=1 后运行真实 MLflow 集成测试",
)


def test_log_artifact_file():
    import mlflow

    from scripts.mlflow_tracking import log_artifact_file

    project_root = Path(__file__).resolve().parents[1]
    local_path = project_root / "README.md"
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("test-demand-forecasting")
    with mlflow.start_run(run_name="artifact-check"):
        log_artifact_file(
            local_path,
            artifact_path="tests",
        )
