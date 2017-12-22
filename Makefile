.PHONY: test

test: easy_abx/utils.py easy_abx/prepare_abx.py easy_abx/run_abx.py
	python easy_abx/utils.py -v 

clean:
	python setup.py clean

dev:
	python setup.py develop 

