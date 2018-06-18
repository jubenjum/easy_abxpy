.PHONY: test

test: easy_abx/utils.py easy_abx/prepare_abx.py easy_abx/run_abx.py
	python easy_abx/utils.py -v 

clean:
	python setup.py clean
	rm articulatory.* SP4050BM_*

dev:
	python setup.py develop 


conda:
	rm -rf outputdir
	conda build --output-folder outputdir -n .
	#conda convert --platform all outputdir/linux-64/*.tar.bz2 -o outputdir/
	for dfile in outputdir/*/*.tar.bz2; do \
		anaconda upload --force -u primatelang $$dfile; \
	done
	    

