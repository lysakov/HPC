bsub -n $1 -R "span[ptile=1]" -gpu "num=1:mode=exclusive_process" -o out -e err mpiexec ./PoissonEquation -m MPI -b CUDA -N $3 -M $2
