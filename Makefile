install:
	pip install --upgrade pip && pip install -r requirements/requirements.txt && pip install -r requirements/test_requirements.txt

install_api:
	pip install -r obesity_model_api/requirements.txt

train_pipeline:
	python obesity_model/train_pipeline.py

format:
	black . *.py

lint:
	pylint --disable=R,C obesity_model/ obesity_model_api/
mypy:
	mypy --implicit-optional obesity_model/ obesity_model_api/ tests/

test:
	python -m pytest

all: install format lint mypy test train_pipeline 