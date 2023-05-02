set terminal png size 1000,600
set xlabel "Test Offset" font ",15"
set ylabel "Network Predict(%)" font ",15"
set yrange [80:100]
set title "One Layer vs Two Layers" font ",15"
set output "layer.png"
set style data linespoints
plot "one_layer_performance_10k.csv" using 4:xtic(1) smooth bezier title "1 Layer" lt rgb "blue", "two_layers_performance_10k.csv" using 4:xtic(1) smooth bezier title "2 Layer" lt rgb "green"

