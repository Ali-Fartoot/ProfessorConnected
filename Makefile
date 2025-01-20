PYTHON = python3
USE_GPU ?= false

.PHONY: install run test cold-start request

install:
	. ./venv/bin/activate &&  pip install -r requirements.txt

run:
	@echo "Starting LLM server..."
	. ./venv/bin/activate && $(PYTHON) -m llama_cpp.server --port 5333  \
		--n_ctx 14000 \
		$(if $(filter true,$(USE_GPU)),--n_gpu_layers 16,--n_gpu_layers 0) \
		--model ./models/Mistral-Nemo-Prism-12B-Q6_K.gguf \
		--chat_format mistral-instruct > llm.log 
	@echo "Starting FastAPI server..."
	. ./venv/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000

test:
	@echo "Running tests..."
	. ./venv/bin/activate && pytest ./tests/ -v --capture=no --log-cli-level=INFO

cold-start:
	. ./venv/bin/activate && $(PYTHON) cold_start.py

request:
	./run.sh

clean:
	@echo "Cleaning data folders"
	rm -rf data/*
	rm -rf professor_db/*
	rm -rf figures