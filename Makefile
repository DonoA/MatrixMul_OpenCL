ARGS := -std=gnu++11 -g -Wall -Wextra

default: matmul_cl

matmult_omp: matmult_omp.cpp
	g++ -fopenmp $(ARGS) -o $@ $<

# module load CUDA/10.1.243-GCC-8.3.0
matmul_cl: matmul_cl.cpp
	g++ -lOpenCL $(ARGS) -o $@ $<

clean:
	-rm matmul_cl