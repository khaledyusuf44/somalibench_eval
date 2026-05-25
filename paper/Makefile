# Paper build -- arXiv-ready source package.
#
#   make figs        refresh PDFs from ../data/figures
#   make pdf         build main.pdf locally if pdflatex/bibtex are installed
#   make zip         build arxiv-submission.zip
#   make clean       remove local LaTeX build products

FIG_PDFS := figures/fig1_refusal_rates.pdf \
            figures/fig2_gap_forest.pdf \
            figures/fig3_per_category.pdf

.PHONY: all figs pdf zip clean

all: zip

figs:
	mkdir -p figures
	cp ../data/figures/fig1_refusal_rates.pdf figures/
	cp ../data/figures/fig2_gap_forest.pdf figures/
	cp ../data/figures/fig3_per_category.pdf figures/

pdf: main.pdf

main.pdf: main.tex refs.bib $(FIG_PDFS)
	pdflatex -interaction=nonstopmode main.tex
	bibtex main || true
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

zip: arxiv-submission.zip

arxiv-submission.zip: main.tex refs.bib $(FIG_PDFS)
	rm -rf _build && mkdir -p _build/figures
	cp main.tex refs.bib _build/
	cp $(FIG_PDFS) _build/figures/
	rm -f arxiv-submission.zip
	cd _build && zip -r ../arxiv-submission.zip . -x "*.DS_Store"
	rm -rf _build

clean:
	rm -f main.aux main.log main.out main.bbl main.blg main.pdf main.synctex.gz
	rm -rf _build
