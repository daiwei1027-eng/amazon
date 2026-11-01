"""真实 MinIO 集成测试：设置 RUN_MINIO_TESTS=1 后运行。"""

import os
from datetime import date
from pathlib import Path
from uuid import uuid4

import polars as pl
import pytest
import urllib3
import yaml
from dotenv import load_dotenv
from minio import Minio

from scripts.ingest_minio_bucket import ingest_bucket


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MINIO_TESTS") != "1",
    reason="设置 RUN_MINIO_TESTS=1 后运行真实 MinIO 集成测试",
)


@pytest.fixture
def minio_upload_target():
    """准备独立测试桶和唯一对象名，测试结束后清理该对象。"""
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    config_path = project_root / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as file:
        minio_config = yaml.safe_load(file)["services"]["minio"]

    client = Minio(
        endpoint=(
            f"{minio_config['host']}:"
            f"{minio_config['ports']['api']}"
        ),
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=minio_config["secure"],
        http_client=urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=3, read=10),
            retries=False,
        ),
    )

    bucket_name = "tests"
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    object_name = f"ingest/{uuid4().hex}/sample.parquet"
    try:
        yield client, bucket_name, object_name
    finally:
        client.remove_object(bucket_name, object_name)


def test_ingest_bucket_roundtrip(tmp_path, minio_upload_target):
    client, bucket_name, object_name = minio_upload_target
    source_path = tmp_path / "source.parquet"
    downloaded_path = tmp_path / "downloaded.parquet"

    pl.DataFrame(
        {
            "item_id": ["A", "B"],
            "date": [date(2026, 10, 27), date(2026, 10, 28)],
            "units_sold": [0, 3],
            "sell_price": [None, 4.34],
        }
    ).write_parquet(source_path)

    ingest_bucket(
        client=client,
        file_path=source_path,
        target_bucket=bucket_name,
        target_path=object_name,
    )
    client.fget_object(bucket_name, object_name, str(downloaded_path))

    assert downloaded_path.read_bytes() == source_path.read_bytes()
