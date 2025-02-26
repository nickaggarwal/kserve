dev_install:
	poetry install --with test -E vllm --no-interaction

install_dependencies:
	poetry install --with=test -E vllm --no-interaction --no-root
	
test: type_check
	pytest -W ignore

type_check:
	mypy --ignore-missing-imports huggingfaceserver 

test_speculative:
	python3 -m huggingfaceserver --speculative-model "path/to/speculative/model" --speculative-model-revision "v1.0"
