
#include "box.hpp"
#include "fmm.hpp"
#include "matrix.hpp"
#include "timer.hpp"
#include <vector>
#include <iostream>

constexpr size_t N = 200000;
constexpr int dim = 3;
using Vector_t = Vector<double, dim>;


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize random positions in a cube [0, 1]^3
    std::vector<Vector_t> positions;
    std::vector<Vector_t> vector_charges;
    positions.reserve(N);
    vector_charges.reserve(N);
    for (size_t i=0; i<N; ++i) {
        positions.push_back({static_cast<double>(std::rand())/RAND_MAX,
                             static_cast<double>(std::rand())/RAND_MAX,
                             static_cast<double>(std::rand())/RAND_MAX});
        vector_charges.push_back({static_cast<double>(std::rand())/RAND_MAX,
                                  static_cast<double>(std::rand())/RAND_MAX,
                                  static_cast<double>(std::rand())/RAND_MAX});
    }

    // Define simulation domain by a cube
    int vertical_axis = 2;
    Box<double, dim, Vector> box(positions, vertical_axis);
    if (rank == 0) {
        std::cout << box << std::endl;
    }

    // Adaptively partition the simulation domain
    int max_level = 20;
    int max_particles_per_node = 128;
    OctreePartitioner<double, Vector> partitioner(positions, box);
    partitioner.refine(max_level, max_particles_per_node);
    if (rank == 0) {
        std::cout << partitioner << std::endl;
    }

    // Precompute FMM
    constexpr int expansion_order = 8;
    FMM3D<double, dim, expansion_order, Vector, true> fmm(partitioner, MPI_COMM_WORLD);

    // Calculation via FMM
    std::vector<Vector_t> vector_potentials_fmm(N);
    std::vector<Vector_t> vector_fields_fmm(N);

    tic("FMM potential calculation");
    fmm.rinv(vector_charges.data(), vector_potentials_fmm.data());
    MPI_Allreduce(MPI_IN_PLACE, vector_potentials_fmm.data(), dim*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    toc("FMM potential calculation");

    tic("FMM field calculation");
    fmm.rinv<true>(vector_charges.data(), vector_fields_fmm.data());
    MPI_Allreduce(MPI_IN_PLACE, vector_fields_fmm.data(), dim*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    toc("FMM field calculation");

    // Calculation via direct method
    svector<Vector_t> vector_potentials_direct(MPI_COMM_WORLD, N);
    svector<Vector_t> vector_fields_direct(MPI_COMM_WORLD, N);

    tic("Direct potential calculation");
    for (size_t i=begin(N, size, rank); i<end(N, size, rank); ++i) {
        for (size_t j=0; j<N; ++j) {
            if (i != j) {
                const auto R = positions[i] - positions[j];
                vector_potentials_direct[i] += vector_charges[j]/R.norm();
            }
        }
    }
    vector_potentials_direct.sync(); vector_potentials_direct.template allreduce<double>();
    toc("Direct potential calculation");

    tic("Direct field calculation");
    for (size_t i=begin(N, size, rank); i<end(N, size, rank); ++i) {
        for (size_t j=0; j<N; ++j) {
            if (i != j) {
                const auto R = positions[i] - positions[j];
                vector_fields_direct[i] += vector_charges[j].cross(R)/std::pow(R.norm(), 3);
            }
        }
    }
    vector_fields_direct.sync(); vector_fields_direct.template allreduce<double>();
    toc("Direct field calculation");

    // Calculate error
    if (rank == 0) {
        double numer_potential = 0;
        double denom_potential = 0;
        double numer_field = 0;
        double denom_field = 0;
        for (size_t i=0; i<N; ++i) {
            numer_potential += std::pow((vector_potentials_direct[i] - vector_potentials_fmm[i]).norm(), 2);
            denom_potential += std::pow(vector_potentials_direct[i].norm(), 2);
            numer_field += std::pow((vector_fields_direct[i] - vector_fields_fmm[i]).norm(), 2);
            denom_field += std::pow(vector_fields_direct[i].norm(), 2);
        }
        std::cout << "Relative l2 error of potential = " << std::sqrt(numer_potential/denom_potential) << std::endl;
        std::cout << "Relative l2 error of field = " << std::sqrt(numer_field/denom_field) << std::endl;
    }

    MPI_Finalize();
}
