install:
	python setup.py install

dev:
	python setup.py develop 

test: easy_abx/utils.py easy_abx/prepare_abx.py easy_abx/run_abx.py
	python easy_abx/utils.py -v 
