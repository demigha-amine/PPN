set terminal png size 1000,600
set xlabel "Test Offset" font ",15"
set ylabel "Network Predict(%)" font ",15"
set yrange [80:100]
set auto x
set grid
set key left
set key box
set title "Sigmoid vs ReLu" font ",15"
set output "sig_relu.png"
set style data linespoints
plot "Relu_60k_offset_performance.csv" using 3:4 title "ReLu" lt rgb "blue", "Relu+Softmax_60k_offset_performance.csv" using 3:4 title "ReLu + Softmax" lt rgb "green", "Sigmoid_60k_offset_performance.csv" using 3:4 title "Sigmoid" lt rgb "red", "Sigmoid+Softmax_60k_offset_performance.csv" using 3:4 title "Sigmoid + Softmax" lt rgb "yellow"

