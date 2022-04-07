RMD := $(wildcard *.rmd)
HTML := $(RMD:.rmd=.html)
PDF := $(RMD:.rmd=.pdf)
TEX := $(RMD:.rmd=.tex)
DOCX := $(RMD:.rmd=.docx)
BIB := DeltaETM.bib

all:  $(PDF) $(TEX) $(DOCX) $(HTML)

clean: 
	rm -f $(PDF) $(TEX) $(DOCX) $(HTML)

%.html: %.rmd $(BIB)
	Rscript -e "rmarkdown::render('$<', rmarkdown::html_document(pandoc_args = '--self-contained'));"

%.pdf: %.rmd $(BIB)
	Rscript -e "rmarkdown::render('$<', 'pdf_document');"

%.tex: %.rmd $(BIB)
	Rscript -e "rmarkdown::render('$<', 'latex_document');"

%.docx: %.rmd $(BIB)
	Rscript -e "rmarkdown::render('$<', 'word_document');"

