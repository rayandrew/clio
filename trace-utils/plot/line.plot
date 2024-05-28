#!/usr/bin/gnuplot --persist

load "colors.plot"

# Arguments
DATA = ARG1
OUTPUT = ARG2
TITLE = ARG3

if (ARGC > 5) {
    X_MIN = ARG4
    X_MAX = ARG5
}

set term postscript eps enhanced color 20
set output sprintf("%s", OUTPUT)
set size 1,.7
set key right bottom
set samples 1000
# set xrange [X_MIN:X_MAX]
set autoscale xfix
set xlabel ""
set ylabel TITLE

plot DATA using 1:2 \
    title "" \
    with lines \
    linecolor rgb COLORS[1] \
    linewidth 7