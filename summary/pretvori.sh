#!/bin/bash

pandoc "$1" \
    -V linkcolor:blue \
    -V geometry:a4paper \
    -V fontsize=12pt \
    -V geometry:margin=2cm \
    -V lang=sl \
    --highlight-style custom_kate.theme \
    --toc -N \
    --include-in-header head.tex \
    --pdf-engine=xelatex \
    -o "$2"

echo "Done!"
read -rn1
