set terminal png size 1000,600
set title "Nodes Performance"
set xlabel "NODES PER HIDDEN"
set ylabel "NETWORK PREDICT %"
set auto x
set yrange[60:100]
set grid
set key left
set key box
set key height 2
set key width 2
set output 'Nodes_Perf.png' 
set style data linespoints
plot 'nodes_performance_0k2k.csv' using 2:4 lt 1 lw 2 title '0k2k', 'nodes_performance_2k4k.csv' using 2:4 lt 2 lw 2 title '2k4k', 'nodes_performance_4k6k.csv' using 2:4 lt 3 lw 2 title '4k6k', 'nodes_performance_6k8k.csv' using 2:4 lt 4 lw 2 title '6k8k'
