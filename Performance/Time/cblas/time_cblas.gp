set terminal png size 1000,600
set title "Time Performance (BASE vs CBLAS)"
set xlabel "TRAINING SIZE"
set ylabel "TIME(s)"
set auto x 
set grid
set key left
set key box
set key height 2
set key width 4
set output 'TIME_CBLAS.png'
set style data linespoints
plot 'Training_size_Time_BASE.csv' using 1:3 lt 1 lw 2 title 'BASE', 'Training_size_Time_CBLAS.csv' using 1:3 lt 2 lw 2 title 'FULL CBLAS'
