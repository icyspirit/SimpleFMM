# SimpleFMM

A simple Fast Multipole Method (FMM) implementation using MPI shared memory and OpenMP SIMD vectorization.  
This is a header-only library and can be easily included in any C++ project.

## Dependencies and Compilation

- C++17
- MPI
- OpenMP

To compile:
```bash
make CC=$YOUR_OWN_CXX_COMPILER
./a.out               # or: mpirun -n $NP ./a.out
```

## Usage
See `main.cpp` for an example of how to use the library.

`-DFMM_M2L_PLANE_WAVE` translates multipole to local through plane waves rather
than by rotation, which is 2.5x faster at `p = 12`.  The quadrature table is
picked from `p`; `-DFMM_EXP_DIGITS=3|6|9` overrides it.

## License
This project is licensed under the MIT License.
