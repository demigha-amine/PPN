set terminal png size 1000,600
set xlabel "Number of Nodes" font ",15"
set ylabel "Network Predict(%)" font ",15"
set yrange [70:100]
set title "Sigmoid vs ReLu" font ",15"
set output "sig_relu.png"
set style data linespoints
plot "relu_performance.csv" using 4:xtic(2) smooth bezier title "ReLu" lt rgb "blue", "sigmoid_performance.csv" using 4:xtic(2) smooth bezier title "Sigmoid" lt rgb "green"

