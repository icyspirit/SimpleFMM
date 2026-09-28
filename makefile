.PHONY: all clean

CXX = mpicxx
CXXFLAGS ?=

all: fmm_test

fmm_test: main.cpp $(wildcard *.hpp) makefile
	$(CXX) -std=c++17 -Wall -Wextra -Wno-missing-braces -fopenmp -O3 -ffast-math -march=native -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX $(CXXFLAGS) -o $@ main.cpp

clean:
	rm -f fmm_test
