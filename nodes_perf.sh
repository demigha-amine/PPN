#!/bin/bash


echo "DATA SET ; NODES ; TRAINING TIME ; NETWORK PREDICT" > nodes_performance.csv

for (( nodes=60; nodes<=1000; nodes+=20 ))
do
    make clean
    make
    ./exe 1000 $nodes >> nodes_performance.csv
done