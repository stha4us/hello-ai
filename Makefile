# base shell
SHELL := /bin/bash

# export env variables
-include .env

# STANDARD COLORS
ifneq (,$(findstring xterm,${TERM}))
	BLACK        := $(shell tput -Txterm setaf 0)
	RED          := $(shell tput -Txterm setaf 1)
	GREEN        := $(shell tput -Txterm setaf 2)
	YELLOW       := $(shell tput -Txterm setaf 3)
	LIGHTPURPLE  := $(shell tput -Txterm setaf 4)
	PURPLE       := $(shell tput -Txterm setaf 5)
	BLUE         := $(shell tput -Txterm setaf 6)
	WHITE        := $(shell tput -Txterm setaf 7)
	RESET := $(shell tput -Txterm sgr0)
else
	BLACK        := ""
	RED          := ""
	GREEN        := ""
	YELLOW       := ""
	LIGHTPURPLE  := ""
	PURPLE       := ""
	BLUE         := ""
	WHITE        := ""
	RESET        := ""
endif

TARGET_COLOR := $(PURPLE)

# VENV variables
VENV_NAME?=.venv
VENV_BIN_PATH?=./$(VENV_NAME)/bin/
VENV_ACTIVATE=$(VENV_BIN_PATH)activate

# PYTHON
PYTHON?=python3

all: venv dev-install ## Creates a new virtual environment and installs the.

refresh-env: clean all ## Regenerates the working enviroment and installs the dependencies from scratch.

venv: ## Creates a new virtual environment using venv, with the latest version of pip.
	(\
	  test -d $(VENV_NAME) || $(PYTHON) -m venv $(VENV_NAME) ;\
	  . $(VENV_ACTIVATE) ;\
	  pip install --upgrade pip ;\
	  pip -V ;\
	  which python ;\
	)

.PHONY: refresh-env venv

dev-install: venv ## Creates a new virtual environment and install base the dependencies on it.
	. $(VENV_ACTIVATE) && (\
	  pip3 install -r requirements.txt ;\
	  $(PYTHON) -m ipykernel install --user --name ${PROJECT_TAG} --display-name ${PROJECT_TAG} ;\
    )

update-deps: venv ## Updates project dependencies.
	. $(VENV_ACTIVATE) && (\
		python -m pip install --upgrade -r ./requirements.txt ;\
	)

create-env: venv ## Creates .env file based on your general local env.
	ENV=$(ENV) envsubst < .env.sample > .env ;

.PHONY: dev-install update-deps create-env

# CLEAN enviroment
clean-venv: ## Removes virtual environment.
	rm -fr $(VENV_NAME)

clean-test: ## Removes testing related files.
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache/

clean: clean-venv clean-test ## Cleans project installation.

.PHONY: clean-venv clean-test clean

# ====== ENVIRONMENT VARIABLE DEFAULTS ======
ENV?=dev

TODAY := $(shell date +%Y-%m-%d)

pipeline: venv
	. $(VENV_ACTIVATE) && (\
	echo 'Starting pipeline...' ;\
	python -m timeseries.cli ingest-features --multiprocessing "$(MULTIPROCESSING)"\
	$(MAKE) forex_forecast ;\
	)

.PHONY: pipeline

# ====== RUN PIPELINE COMMANDS ======
forex_forecast:
	python forex/model/timeseries/ingest.py
	python forex/model/timeseries/train.py 
	python forex/model/timeseries/predict.py
	python forex/model/timeseries/report.py

.PHONY: forex_forecast

# ====== @TODO DOCKER ======

# ====== HELP ======

help: ## Show this help.
	@grep -E '^[a-zA-Z_0-9%-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "${TARGET_COLOR}%-30s${RESET} %s\n", $$1, $$2}'

.PHONY: help


