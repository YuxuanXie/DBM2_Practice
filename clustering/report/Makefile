
ifeq ($(UNAME_S), Darwin)
	EDITOR = open
else
	EDITOR = evince 
endif

all:
	pdflatex report
	$(EDITOR) report.pdf
