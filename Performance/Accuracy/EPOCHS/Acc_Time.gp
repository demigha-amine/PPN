set terminal png size 1000,600
set title "Accuracy Over Time Performance"
set xlabel "EPOCHS"
set ylabel "NETWORK PREDICT %"
set auto x 
set grid
set key bottom right
set key box
set key height 2
set key width 2
set output 'Acc_Time_Perf.png'
set style data linespoints
plot 'Acc_time_10k.csv' using 1:2 lt 1 lw 2 smooth bezier title 'RelSoft'
