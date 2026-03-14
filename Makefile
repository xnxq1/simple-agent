
lint: ## Проверить код с помощью ruff
	ruff check .

lint-fix: ## Проверить и автоисправить код
	ruff check . --fix

format: ## Форматировать код с помощью ruff
	@ruff check --select F401 --fix $(git diff --name-only --diff-filter=d $(git merge-base HEAD "origin/master") | grep -E "\.py$") 2>/dev/null || true
	@ruff format $(git diff --name-only --diff-filter=d $(git merge-base HEAD "origin/master") | grep -E "\.py$")
	@ruff check --select I --fix $(git diff --name-only --diff-filter=d $(git merge-base HEAD "origin/master") | grep -E "\.py$") 2>/dev/null || true
	@ruff check --select E301,E302,E303,E305 --fix $(git diff --name-only --diff-filter=d $(git merge-base HEAD "origin/master") | grep -E "\.py$") 2>/dev/null || true

eval: ## Оценить систему RAG с пороговыми значениями
	mkdir -p reports
	python scripts/eval_rag.py \
		--base-url http://localhost:8000 \
		--min-context-score 0.6 \
		--min-faithfulness 0.7 \
		--min-answer-relevance 0.65 \
		--min-abstain-rate 0.7 \
