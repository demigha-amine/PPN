#!/bin/bash


echo "DATA SET ; NODES ; TRAINING TIME ; NETWORK PREDICT" > size_performance.csv

for (( size=250; size<=10000; size+=250 ))
do
    make clean
    make
    ./exe $size 560 >> size_performance.csv
done

