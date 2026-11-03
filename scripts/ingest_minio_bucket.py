import os
from dotenv import load_dotenv
from minio import Minio
from pathlib import Path
import yaml

def ingest_bucket(
    client: Minio,
    file_path: Path,
    target_bucket: str,
    target_path: str,
):
    return client.fput_object(target_bucket, object_name=target_path, file_path=str(file_path))

def main():
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    config_path = project_root / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    bucket = config["services"]["minio"]["bucket"]
    prefix = config["services"]["minio"]["prefix"]
    data_config = config["data"]
    output_config = config["features"]["directory"]
    filename=config["features"]["filename"]
    file_path = (
            project_root
            / data_config["directory"]
            / output_config
            / filename
    )
    target_path = f"{prefix}/{file_path.name}"
    minio_config = config["services"]["minio"]

    client = Minio(
        endpoint=(
            f"{minio_config['host']}:"
            f"{minio_config['ports']['api']}"
        ),
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=minio_config["secure"],
    )
    result = ingest_bucket(client, file_path, bucket, target_path)
    print(f"Uploaded: {result.bucket_name}/{result.object_name}")
if __name__ == "__main__":
    main()