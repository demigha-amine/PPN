set terminal png size 1000,600
set title "Activation Functions Performance" font ",13"
set xlabel "TEST OFFSET" font ",10"
set ylabel "NETWORK PREDICT (%)" font ",10"
set auto x
set yrange[50:100]
set grid
set key left
set key box
set output 'Activation_Perfs.png'
set style data linespoints
plot 'Relu_60k_offset_performance.csv' using 3:4 lt 1 lw 2 title 'Relu', 'Sigmoid_60k_offset_performance.csv' using 3:4 lt 2 lw 2 title 'Sigmoid', 'Relu+Softmax_60k_offset_performance.csv' using 3:4 lt 3 lw 2 title 'Relu+Softmax', 'Sigmoid+Softmax_60k_offset_performance.csv' using 3:4 lt 4 lw 2 title 'Sigmoid+Softmax'

