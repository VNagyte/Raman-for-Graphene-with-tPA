# Welcome to Raman spectroscopy analysis tool for graphene

Raman spectroscopy analysis script created to evaluate Raman parameters of graphene and detect trans-polyacetylene chains. By performing individual Raman peak fittings of two areas: around G and 2D peaks, this makes it is possible consistently ectract information and analyse it. Additionally, the presence of trans-polyacetelyne chains evaluated. One has to remember that for accurate analysis, the Raman spectrum for G peak should start from 1050 cm<sup>-1<sup>.

# How to proceed

1. Edit Python script according to your data:
    1. Rename file file_baseg, file_base2d depending on your .txt file names.
    1. Set first_file number.
    1. Provide with ratio between intensities of G and 2D peaks.
1. Place the script into a folder with all data.
1. Run it.
1. All results will be in the sample folder: figures (.png), fitting parameters (.cvs), normalised spectra (.txt).
