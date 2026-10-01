install:
	pip install .

install-dev:
	pip install -e ".[dev]"

.PHONY: black
black:
	black src/living

.PHONY: black-check
black-check:
	black --check src/living

.PHONY: all clean install install-dev black black-check