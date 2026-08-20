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

Multipole to local translation goes through plane waves, which is 2 to 2.8x
faster than the rotation form.  `-DFMM_M2L_ROTATION` selects the rotation form
instead.  The quadrature table is picked from `p`; `-DFMM_EXP_DIGITS=3|6|9`
overrides it.

## License
This project is licensed under the MIT License.
