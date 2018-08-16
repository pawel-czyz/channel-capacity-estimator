test:
	@echo "Running unit tests..."
	@python3 -m unittest discover -s tests > /dev/null

coverage:
	@echo "Checking test coverage using unittest module..."
	@coverage run --source=cce -m unittest discover -s tests > /dev/null; coverage report

html:
	@echo "Checking test coverage and showing the results in browser..."
	@coverage run --source=cce -m unittest discover -s tests > /dev/null; coverage html; python -m webbrowser "./htmlcov/index.html" &

comp:
	@echo "Checking compatibility with Python 2.7..."
	@python2 -m unittest discover -s tests > /dev/null

install:
	@echo "Installing cce module via pip..."
	@pip install .

.PHONY: test coverage html comp install
