
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
		sudo apt-get install libsndfile1

test:
	python -m pytest -vv test_hello.py

format:
	black *.py

lint:
	pylint extract_data.py

all: install lint test