ARGS := -std=c++11 -g -Wall -Wextra

default: matmul_cl

matmul_cl: matmul_cl.cpp
	g++ $(ARGS) -o $@ $<
