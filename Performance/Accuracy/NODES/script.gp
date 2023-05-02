set terminal png size 1000,600
set xlabel "Number of Nodes" font ",15"
set ylabel "Network Predict(%)" font ",15"
set yrange [80:100]
set auto x
set grid
set title "Nodes Performance" font ",15"
set output "nodes_perf.png"
set style data linespoints
plot "nodes_performance_0k2k.csv" using 2:4 title "0k-2k" lt rgb "blue", "nodes_performance_2k4k.csv" using 2:4 title "2k-4k" lt rgb "green", "nodes_performance_4k6k.csv" using 2:4 title "4k-6k" lt rgb "red", "nodes_performance_6k8k.csv" using 2:4 title "6k-8k" lt rgb "yellow"

