.DEFAULT_GOAL := help
.PHONY: config up down status logs train test health help
help:
	@echo Available commands:
	@echo   make config  - Validate Docker Compose configuration
	@echo   make up      - Start services
	@echo   make down    - Stop services
	@echo   make status  - Show service status
	@echo   make logs    - Show MLflow logs
	@echo   make train   - Run training or baseline evaluation
	@echo   make test    - Run ordinary tests
	@echo   make health  - Check service endpoints
	@echo   make help    - Show this help
config:
	docker compose config --quiet

up:
	docker compose up -d

down:
	docker compose down

status:
	docker compose ps

logs:
	docker compose logs --tail 50 mlflow

train:
	python -m scripts.train

test:
	python -m pytest tests -q
health:
	@echo Checking MLflow...
	@curl.exe -fsS --connect-timeout 2 --max-time 5 -o NUL http://127.0.0.1:5000/health
	@echo Checking MinIO API...
	@curl.exe -fsS --connect-timeout 2 --max-time 5 -o NUL http://127.0.0.1:9000/minio/health/ready
	@echo Checking MinIO Console...
	@curl.exe -fsS --connect-timeout 2 --max-time 5 -o NUL http://127.0.0.1:9001/
	@echo All health checks passed.