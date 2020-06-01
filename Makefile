ARGS := -lOpenCL -std=c++11 -g -Wall -Wextra

default: matmul_cl

# module load CUDA/10.1.243-GCC-8.3.0
matmul_cl: matmul_cl.cpp
	g++ $(ARGS) -o $@ $<

clean:
	-rm matmul_cl