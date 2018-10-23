SHELL=/bin/bash
FIGURES := figures/capd-pupil-deconv-attn-by-space-group.pdf \
           figures/capd-pupil-deconv-group-by-space-attn.pdf \
           figures/trial-diagram.pdf
SOURCETABLES := tables/table-accuracy.tex tables/table-rt.tex \
	        	tables/table-ttest.tex
TABLES := table-accuracy.tex table-rt.tex table-ttest.tex
TEMPFILES := tmp.md tmp.tex tmp2.tex tmp.pdf
CLASSFILES := JASAnew.cls JASA-EL.cls jasaauthyear2.bst
EXTS := .aux .bbl .blg .log .ent Notes.bib

.PHONY: ascii classlink elclasslink biblink tablelink clean tables upload R1

ascii: manuscript.md
	@python templates/asciify.py manuscript.md tmp.md
	@python templates/detect-non-ascii.py tmp.md

classlink:
	@ln -s templates/JASAnew.cls

elclasslink:
	@ln -s templates/JASA-EL.cls

biblink:
	@ln -s bib/jasaauthyear2.bst

tablelink:
	@for table in $(SOURCETABLES); do ln -s $$table; done

tables: $(SOURCETABLES)
	@cd tables; python jasaify-tables.py $(TABLES)

clean:
	@rm -f nonascii.txt $(CLASSFILES) $(TEMPFILES) $(TABLES)
	@for ext in $(EXTS); do \
		rm -f tmp$$ext; \
		rm -f manuscript-submission$$ext; done

supplement: tablelink supplement.md
	@pandoc supplement.md \
	--from markdown-smart \
	--template=templates/template-draft.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--variable documentclass=article \
	--variable classoption=oneside \
	--variable geometry:letterpaper \
	--variable geometry:margin=2.5cm \
	--variable fontsize=12pt \
	--output=manuscript-supplement.pdf
	@rm $(TABLES)

draft: tablelink manuscript.md $(FIGURES)
	@pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=templates/template-draft.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--variable documentclass=article \
	--variable classoption=oneside \
	--variable geometry:letterpaper \
	--variable geometry:margin=2.5cm \
	--variable fontsize=12pt \
	--csl bib/jasa.csl \
	--output=manuscript-draft.pdf
	@rm $(TABLES)

reprint: ascii classlink biblink tables manuscript.md $(FIGURES)
	@pandoc tmp.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--template=templates/template-jasa.tex \
	--pdf-engine=xelatex \
	--number-sections \
	--variable documentclass=JASAnew \
	--variable classoption=reprint \
	--variable classoption=12pt \
	--variable biblio-style=jasaauthyear2.bst \
	--output=manuscript-preprint.pdf
	@rm -f $(CLASSFILES) $(TEMPFILES) $(TABLES)

submission: classlink biblink tables manuscript.md $(FIGURES)
	# Making LaTeX source...
	@cp manuscript.md tmp.md  # do this if not using asciify
	@pandoc tmp.md \
	--from=markdown-smart \
	--template=templates/template-jasa.tex \
	--number-sections \
	--variable documentclass=JASAnew \
	--variable classoption=preprint \
	--variable classoption=authaffil \
	--variable classoption=trackchanges \
	--variable biblio-style=jasaauthyear2.bst \
	--natbib \
	--output=tmp.tex
	# Making BBL...
	@pdflatex tmp.tex
	@bibtex tmp.aux
	@python templates/input-bbl.py tmp.tex tmp.bbl tmp2.tex
	@latexpand --keep-comments tmp2.tex -o manuscript-submission.tex
	# Compiling...
	@pdflatex manuscript-submission.tex
	@pdflatex manuscript-submission.tex
	@pdflatex manuscript-submission.tex
	# Cleaning up...
	@rm -f $(CLASSFILES) $(TEMPFILES) $(TABLES)
	@for ext in $(EXTS); do \
		rm -f tmp$$ext; \
		rm -f manuscript-submission$$ext; done
	# make sure we caught all non-ASCII glyphs
	@python templates/detect-non-ascii.py manuscript-submission.tex

R1: submission response
	@rm -rf R1
	@mkdir -p R1
	@cp manuscript-supplement.pdf R1/SuppPub1.pdf
	@cp figures/trial-diagram.pdf R1/Figure1.pdf
	@cp figures/accuracy-figure-probit.pdf R1/Figure2.pdf
	@cp figures/rt-figure.pdf R1/Figure3.pdf
	@cp figures/capd-pupil-deconv-group-by-space-attn.pdf R1/Figure4.pdf
	@cp figures/capd-pupil-deconv-attn-by-space-group.pdf R1/Figure5.pdf
	@cp manuscript-submission.pdf R1/manuscript.pdf
	@python templates/fix-figure-includes.py manuscript-submission.tex \
		R1/manuscript.tex
	@cp response-to-reviewers.pdf R1/

response:
	@pandoc response-to-reviewers.md \
	--from markdown-smart \
	--template=templates/template-response.tex \
	--pdf-engine=xelatex \
	--output=response-to-reviewers.pdf


upload:
	@rm -rf upload
	@mkdir -p upload
	@cp manuscript-supplement.pdf upload/SuppPub1.pdf
	@cp figures/trial-diagram.pdf upload/Figure1.pdf
	@cp figures/accuracy-figure-probit.pdf upload/Figure2.pdf
	@cp figures/rt-figure.pdf upload/Figure3.pdf
	@cp figures/capd-pupil-deconv-group-by-space-attn.pdf upload/Figure4.pdf
	@cp figures/capd-pupil-deconv-attn-by-space-group.pdf upload/Figure5.pdf
	@cp manuscript-submission.pdf upload/manuscript.pdf
	@python templates/fix-figure-includes.py manuscript-submission.tex \
		upload/manuscript.tex
	@cp cover-letter.pdf upload/

coverletter:
	@cd templates; \
	pandoc ../cover-letter.md \
	--from markdown-smart \
	--template=template-ilabs-letterhead.tex \
	--pdf-engine=xelatex \
	--output=../cover-letter.pdf

wordcount:
	@pandoc manuscript.md \
	--from markdown-smart \
	--filter pandoc-citeproc \
	--csl bib/jasa.csl \
	--standalone \
	--mathml \
	--quiet \
	| html2text | wc -w

figures/capd-pupil-deconv-attn-by-space-group.pdf:
	@cd ../..; python plot-pupil-attn-by-space-group.py

figures/capd-pupil-deconv-group-by-space-attn.pdf:
	@cd ../..; python plot-pupil-group-by-space-attn.py

figures/trial-diagram.pdf:
	@cd ../..; python plot-trial-diagram.py

figures/rt-figure.pdf:
	@cd ../..; python plot-rt.py

figures/accuracy-figure-probit.pdf:
	@cd ../..; python plot-accuracy.py
