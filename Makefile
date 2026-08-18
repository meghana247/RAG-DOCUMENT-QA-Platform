.PHONY: help up down logs migrate revision test lint fmt eval shell

help:
	@grep -E '^[a-z-]+:.*?# .*$$' $(MAKEFILE_LIST) | sed 's/:.*# /\t/'

up:            # start the stack
	docker compose up --build -d
	docker compose exec api alembic upgrade head

down:          # stop the stack and remove volumes
	docker compose down -v

logs:          # tail application logs
	docker compose logs -f api

migrate:       # apply migrations
	alembic upgrade head

revision:      # autogenerate a migration: make revision m="add x"
	alembic revision --autogenerate -m "$(m)"

test:          # run the test suite
	pytest -q --cov=app --cov-report=term-missing

lint:          # static checks
	ruff check app tests eval
	mypy app

fmt:           # apply formatting and import sorting
	ruff check --fix app tests eval
	ruff format app tests eval

eval:          # score retrieval against the golden set
	python -m eval.run_eval

shell:         # psql into the database
	docker compose exec db psql -U postgres -d ragdb
