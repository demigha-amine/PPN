set terminal png size 1000,600
set title "Hidden Layers Performance"
set xlabel "TEST OFFSET"
set ylabel "NETWORK PREDICT %"
set auto x 
set yrange[90:100]
set grid
set key left
set key box
set key height 2
set key width 4
set output 'Layers_Perfs.png'
set style data linespoints
plot 'Relu+Softmax_60k_offset_performance.csv' using 3:4 lt 1 lw 2 title '1 LAYER', 'Two_RelSoft_60k_offset_perfs.csv' using 3:4 lt 2 lw 2 title '2 LAYERS'
