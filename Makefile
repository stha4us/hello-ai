.PHONY: build
build:
	docker build . -t timeseries_forex

.PHONY: run
run:
	docker run --env-file .env -it -v "$$(pwd):/home/ec2-user/TIMESERIES_FOREX" timeseries_forex bash

.PHONY: build-and-run
build-and-run: build run

.PHONY: clean
clean: docker system prune

.PHONY: timeseries_forex
forex_forecast:
	python forex/api/model/timeseries/run_ingest.py
	python forex/api/model/timeseries/run_train.py 
	python forex/api/model/timeseries/run_model_predict.py
	python forex/api/model/timeseries/run_report.py
/