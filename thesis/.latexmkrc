$out_dir = 'build';
$aux_dir = 'build';
$pdf_mode = 1;

$ENV{'TEXMF_OUTPUT_DIRECTORY'} = 'C:/Users/andre/Documents/BA/thesis/build';

# Engines (all with shell-escape)
$pdflatex = 'pdflatex -shell-escape -synctex=1 -interaction=errorstopmode %O %S';
$xelatex  = 'xelatex  -shell-escape -synctex=1 -interaction=errorstopmode %O %S';
$lualatex = 'lualatex -shell-escape -synctex=1 -interaction=errorstopmode %O %S';

# Force usage of Biber (instead of BibTeX)
$bibtex_use = 2;
$bibtex = 'biber %O %B';