#!/usr/bin/gnuplot --persist

load "colors.plot"

# Arguments
DATA_DIR = ARG1
OUTPUT = ARG2
PATTERN_EXISTS = 0

if (ARGC >= 3) {
    DATA_PATTERN = ARG3
    print "Plotting data matching ", DATA_PATTERN
    PATTERN_EXISTS = 1
    DATA_=system(sprintf("find %s -type f -name '%s' | sort", DATA_DIR, DATA_PATTERN))
    N_DATA = words(DATA_)
    array DATA[N_DATA]
    do for [i=1:words(DATA_)] {
        DATA[i] = word(DATA_, i)
    }
} else {
    N_DATA = 1
    array DATA[N_DATA]
    DATA[1] = DATA_DIR
}


print "Number of data files: ", N_DATA

## Parse the labels
array LABELS[N_DATA]
array LABELS_DURATION[N_DATA]
do for [i=1:N_DATA] {
    file = DATA[i]
    if (PATTERN_EXISTS) {
        label = system(sprintf("echo %s | sed 's@%s/@@' | sed 's/\.[^.]*$//'", file, DATA_DIR))
    } else {
        label = system(sprintf("basename %s | sed 's/\.[^.]*$//'", file))
    }
    label = system(sprintf("echo %s | sed 's/_/ /g'", label))
    LABELS[i] = label
    LABELS_DURATION[i] = system(sprintf("parse-time %s", label))
}

# sorting data and labels based on duration
do for [i=1:N_DATA] {
    do for [j=i+1:N_DATA] {
        if (LABELS_DURATION[i] > LABELS_DURATION[j]) {
            tmp = LABELS[i]
            LABELS[i] = LABELS[j]
            LABELS[j] = tmp

            tmp = LABELS_DURATION[i]
            LABELS_DURATION[i] = LABELS_DURATION[j]
            LABELS_DURATION[j] = tmp

            tmp = DATA[i]
            DATA[i] = DATA[j]
            DATA[j] = tmp
        }
    }
}

set term postscript eps enhanced color 20
set output sprintf("%s", OUTPUT)
set size 1,.7
set key right bottom
# set xrange
# set xrange [0:]
# set yrange [0:]
set samples 1000
set xlabel ""
set ylabel "IOPS"

plot for [i=1:N_DATA] DATA[i] using 1:2 \
    title LABELS[i] \
    with lines \
    linecolor rgb COLORS[i % NUM_COLORS] \
    linewidth 7