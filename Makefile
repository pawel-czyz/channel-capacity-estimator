# Run unit tests
test:
	python3 -m unittest discover -s tests > /dev/null

# Check test coverage using unittest module
coverage:
	coverage run --source=cce -m unittest discover -s tests > /dev/null; coverage report

# Check test coverage and show the results in browser
html:
	coverage run --source=cce -m unittest discover -s tests > /dev/null; coverage html; python -m webbrowser "./htmlcov/index.html" &

# Check compatibility with Python 2.7
comp:
	python2 -m unittest discover -s tests > /dev/null

install:
	pip install .

.PHONY: test
