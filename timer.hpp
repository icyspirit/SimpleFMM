
#ifndef TIMER_HPP
#define TIMER_HPP


#include <mpi.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <unordered_map>


using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

inline std::unordered_map<std::string, high_resolution_clock::time_point> _map;


inline void tic(const std::string& name) noexcept
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << "Starting " << name << std::endl;
        _map[name] = high_resolution_clock::now();
    }
}

inline void toc(const std::string& name) noexcept
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        assert(_map.find(name) != _map.cend());
        std::cout << "Elapsed time for " << name << ": " << 1e-9*duration_cast<nanoseconds>(high_resolution_clock::now() - _map.at(name)).count() << " [s]" << std::endl;
        _map.erase(name);
    }
}


#endif

