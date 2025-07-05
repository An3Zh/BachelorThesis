$out_dir = 'build';
$aux_dir = 'build';
$pdf_mode = 1;

# Use pdflatex with basic flags
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode %O %S';

# Force usage of Biber (instead of BibTeX)
$bibtex_use = 2;
$bibtex = 'biber %O %B';