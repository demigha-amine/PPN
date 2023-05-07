set terminal png size 1000,600
set title "Time Performance (CBLAS+FLOAT vs OPENMP)"
set xlabel "TRAINING SIZE"
set ylabel "TIME(s)"
set auto x 
set grid
set key left
set key box
set key height 2
set key width 4
set output 'TIME_OPENMP.png'
set style data linespoints
plot 'Training_size_Time_FLOAT.csv' using 1:3 lt 1 lw 2 title 'CBLAS + FLOAT', 'Training_size_Time_6OMP.csv' using 1:3 lt 2 lw 2 title 'OPENMP 6 THREADS', 
