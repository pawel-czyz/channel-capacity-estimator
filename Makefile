# Run unit tests
test:
	python3 -m unittest discover -s tests

# Check test coverage using unittest module
coverage:
	coverage run --source=cce -m unittest discover -s tests; coverage report

# Check test coverage and show the results in browser
html:
	coverage run --source=cce -m unittest discover -s tests; coverage html; python -m webbrowser "./htmlcov/index.html" &

# Check compatibility with Python 2.7
comp:
	python2 -m unittest discover -s tests

.PHONY: test
