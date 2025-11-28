mpicxx -fopenmp heat_diffusion_sync.cpp -o heat_diffusion_sync
mpicxx -fopenmp heat_diffusion_async.cpp -o heat_diffusion_async
mpicxx -fopenmp radioactive_diffusion_sync.cpp -o radioactive_diffusion_sync
mpicxx -fopenmp radioactive_diffusion_async.cpp -o radioactive_diffusion_async
mpicxx -fopenmp shockwave_sync.cpp -o shockwave_sync
mpicxx -fopenmp shockwave_async.cpp -o shockwave_async

mpirun -np 4 --allow-run-as-root \
 --hostfile /root/quoc-huy/lab4-parallel-computing/hosts.txt \
 -x OMP_NUM_THREADS=8 \
 --wd /root/quoc-huy/lab4-parallel-computing \
  ./heat_diffusion_sync
