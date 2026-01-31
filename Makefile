
lint: ## Проверить код с помощью ruff
	ruff check .

lint-fix: ## Проверить и автоисправить код
	ruff check . --fix

format: ## Форматировать код с помощью ruff
	@ruff check --select F401 --fix $(git diff --name-only --diff-filter=d $(git merge-base HEAD "origin/master") | grep -E "\.py$") 2>/dev/null || true
	@ruff format $(git diff --name-only --diff-filter=d $(git merge-base HEAD "origin/master") | grep -E "\.py$")
	@ruff check --select I --fix $(git diff --name-only --diff-filter=d $(git merge-base HEAD "origin/master") | grep -E "\.py$") 2>/dev/null || true
	@ruff check --select E301,E302,E303,E305 --fix $(git diff --name-only --diff-filter=d $(git merge-base HEAD "origin/master") | grep -E "\.py$") 2>/dev/null || true

