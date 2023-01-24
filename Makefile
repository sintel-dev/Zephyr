.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

# CLEAN TARGETS

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-docs
clean-docs: ## remove previously built docs
	rm -f docs/api/*.rst
	rm -rf docs/tutorials
	-$(MAKE) -C docs clean 2>/dev/null  # this fails if sphinx is not yet installed

.PHONY: clean-coverage
clean-coverage: ## remove coverage artifacts
	rm -f .coverage
	rm -f .coverage.*
	rm -fr htmlcov/

.PHONY: clean-test
clean-test: ## remove test artifacts
	rm -fr .tox/
	rm -fr .pytest_cache

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-coverage clean-docs ## remove all build, test, coverage, docs and Python artifacts

# INSTALL TARGETS

.PHONY: install
install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .

.PHONY: install-test
install-test: clean-build clean-pyc ## install the package and test dependencies
	pip install .[test]

.PHONY: install-develop
install-develop: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e .[dev]

# LINT TARGETS

.PHONY: lint
lint: ## check style with flake8 and isort
	flake8 zephyr_ml tests
	isort -c --recursive zephyr_ml tests

.PHONY: fix-lint
fix-lint: ## fix lint issues using autoflake, autopep8, and isort
	find zephyr_ml tests -name '*.py' | xargs autoflake --in-place --ignore-init-module-imports --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive zephyr_ml tests
	isort --apply --atomic --recursive zephyr_ml tests

# TEST TARGETS

.PHONY: test-unit
test-unit: ## run tests quickly with the default Python
	python -m pytest --cov=zephyr_ml

.PHONY: test-readme
test-readme: ## run the readme snippets
	rm -rf tests/readme_test && mkdir -p tests/readme_test/notebooks
	cp -r notebooks/data tests/readme_test/notebooks/
	cd tests/readme_test && rundoc run --single-session python3 -t python3 ../../README.md
	rm -rf tests/readme_test


.PHONY: test-tutorials
test-tutorials: ## run the tutorial notebooks
	find notebooks -path "*/.ipynb_checkpoints" -prune -false -o -name "*.ipynb" -exec \
		jupyter nbconvert --execute --ExecutePreprocessor.timeout=3600 --to=html --stdout {} > /dev/null \;


.PHONY: test
test: test-unit test-readme test-tutorials ## test everything that needs test dependencies

.PHONY: check-dependencies
check-dependencies: ## test if there are any broken dependencies
	pip check

.PHONY: test-devel
test-devel: check-dependencies lint docs ## test everything that needs development dependencies

.PHONY: test-all
test-all:
	tox -r

.PHONY: coverage
coverage: ## check code coverage quickly with the default Python
	coverage run --source zephyr_ml -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

# DOCS TARGETS

.PHONY: docs
docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	sphinx-apidoc --separate --no-toc -o docs/api/ zephyr_ml
	$(MAKE) -C docs html

.PHONY: view-docs
view-docs: docs ## view docs in browser
	$(BROWSER) docs/_build/html/index.html

.PHONY: serve-docs
serve-docs: view-docs ## compile the docs watching for changes
	watchmedo shell-command -W -R -D -p '*.rst;*.md' -c '$(MAKE) -C docs html' docs

# RELEASE TARGETS

.PHONY: dist
dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

.PHONY: publish-confirm
publish-confirm:
	@echo "WARNING: This will irreversibly upload a new version to PyPI!"
	@echo -n "Please type 'confirm' to proceed: " \
		&& read answer \
		&& [ "$${answer}" = "confirm" ]

.PHONY: publish
publish: dist publish-confirm ## package and upload a release
	twine upload --repository-url https://pypi.dailab.ml:8080 dist/*

.PHONY: bumpversion-release
bumpversion-release: ## Merge main to stable and bumpversion release
	git checkout stable || git checkout -b stable
	git merge --no-ff main -m"make release-tag: Merge branch 'main' into stable"
	bumpversion release
	git push --tags origin stable

.PHONY: bumpversion-patch
bumpversion-patch: ## Merge stable to main and bumpversion patch
	git checkout main
	git merge stable
	bumpversion --no-tag patch
	git push

.PHONY: bumpversion-minor
bumpversion-minor: ## Bump the version the next minor skipping the release
	bumpversion --no-tag minor

.PHONY: bumpversion-major
bumpversion-major: ## Bump the version the next major skipping the release
	bumpversion --no-tag major

.PHONY: bumpversion-revert
bumpversion-revert: ## Undo a previous bumpversion-release
	git checkout main
	git branch -D stable

.PHONY: bumpversion-candidate
bumpversion-candidate: ## Bump the version to the next candidate
	bumpversion candidate --no-tag

CLEAN_DIR := $(shell git status --short | grep -v ??)
CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
CHANGELOG_LINES := $(shell git diff HEAD..origin/stable HISTORY.md 2>&1 | wc -l)

.PHONY: check-main
check-main: ## Check if we are in main branch
ifneq ($(CURRENT_BRANCH),main)
	$(error Please make the release from main branch\n)
endif

.PHONY: check-history
check-history: ## Check if HISTORY.md has been modified
ifeq ($(CHANGELOG_LINES),0)
	$(error Please insert the release notes in HISTORY.md before releasing)
endif

.PHONY: check-release
check-release: check-main check-history ## Check if the release can be made

.PHONY: release
release: check-release bumpversion-release docker-push publish bumpversion-patch

.PHONY: release-candidate
release-candidate: check-main publish bumpversion-candidate

.PHONY: release-minor
release-minor: check-release bumpversion-minor release

.PHONY: release-major
release-major: check-release bumpversion-major release

# DOCKER TARGETS

.PHONY: docker-build
docker-build:
	docker build -f docker/Dockerfile -t zephyr_ml .

.PHONY: docker-push
docker-push: docker-build
	@$(eval VERSION := $(shell python -c 'import zephyr_ml; print(zephyr_ml.__version__)'))
	docker tag zephyr_ml docker.pkg.github.com/signals-dev/zephyr_ml/zephyr_ml:$(VERSION)
	docker push docker.pkg.github.com/signals-dev/zephyr_ml/zephyr_ml:$(VERSION)
	docker tag zephyr_ml docker.pkg.github.com/signals-dev/zephyr_ml/zephyr_ml
	docker push docker.pkg.github.com/signals-dev/zephyr_ml/zephyr_ml
