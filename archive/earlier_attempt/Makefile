.PHONY: reproduce assets test check smoke
reproduce:
	python scripts/reproduce.py
assets:
	python -m src.v3.visuals
	python -m src.v3.extra_plots
	python -m src.v3.report
	python -m src.v3.release
	python -m src.v3.deck
	python scripts/export_pdfs.py
test:
	python -m unittest discover -s tests -v
check:
	python scripts/check_consistency.py
smoke:
	python -c 'from src.v3.experiments import validity; validity()'
