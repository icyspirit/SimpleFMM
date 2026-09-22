
#ifndef DISTRIBUTED_HPP
#define DISTRIBUTED_HPP


#include "mpi_util.hpp"
#include <mpi.h>
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>


template<typename I>
inline I begin(I n, int size, int rank) noexcept
{
    return n*rank/size;
}


template<typename I>
inline I end(I n, int size, int rank) noexcept
{
    return n*(rank + 1)/size;
}


template<typename I>
inline I local(I n, int size, int rank) noexcept
{
    return end(n, size, rank) - begin(n, size, rank);
}


inline void allgatherv_inplace(long long count, MPI_Datatype type, void* recvbuf, MPI_Comm comm)
{
    int size;
    MPI_Comm_size(comm, &size);

    MPI_Aint lb, extent;
    MPI_Type_get_extent(type, &lb, &extent);

    std::vector<long long> counts(size);
    MPI_Allgather(&count, 1, MPI_LONG_LONG, counts.data(), 1, MPI_LONG_LONG, comm);

    std::vector<long long> offsets(size + 1, 0);
    for (int rank=0; rank<size; ++rank) {
        offsets[rank + 1] = offsets[rank] + counts[rank];
    }

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (long long base=0; base<std::max(offsets[size], 1LL); base+=INT_MAX) {
        const long long stop = std::min(offsets[size], base + INT_MAX);
        for (int rank=0; rank<size; ++rank) {
            const long long from = std::max(base, offsets[rank]);
            const long long to = std::min(stop, offsets[rank + 1]);
            recvcounts[rank] = to > from ? static_cast<int>(to - from) : 0;
            displs[rank] = to > from ? static_cast<int>(from - base) : 0;
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, static_cast<char*>(recvbuf) + base*extent,
                       recvcounts.data(), displs.data(), type, comm);
    }
}


inline void gatherv_inplace(const void* sendbuf, long long count, MPI_Datatype type, void* recvbuf, int root, MPI_Comm comm)
{
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    MPI_Aint lb, extent;
    MPI_Type_get_extent(type, &lb, &extent);

    std::vector<long long> counts(size);
    MPI_Allgather(&count, 1, MPI_LONG_LONG, counts.data(), 1, MPI_LONG_LONG, comm);

    std::vector<long long> offsets(size + 1, 0);
    for (int r=0; r<size; ++r) {
        offsets[r + 1] = offsets[r] + counts[r];
    }

    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    for (long long base=0; base<std::max(offsets[size], 1LL); base+=INT_MAX) {
        const long long stop = std::min(offsets[size], base + INT_MAX);
        for (int r=0; r<size; ++r) {
            const long long from = std::max(base, offsets[r]);
            const long long to = std::min(stop, offsets[r + 1]);
            recvcounts[r] = to > from ? static_cast<int>(to - from) : 0;
            displs[r] = to > from ? static_cast<int>(from - base) : 0;
        }

        const long long sent = std::max(base, offsets[rank]) - offsets[rank];
        MPI_Gatherv(rank == root ? MPI_IN_PLACE
                                 : static_cast<const char*>(sendbuf) + (recvcounts[rank] > 0 ? sent : 0)*extent,
                    recvcounts[rank], type,
                    rank == root ? static_cast<char*>(recvbuf) + base*extent : nullptr,
                    recvcounts.data(), displs.data(), type, root, comm);
    }
}


template<typename T>
class svector {
private:
    MPI_Comm _intracomm = MPI_COMM_NULL;
    MPI_Comm _intercomm = MPI_COMM_NULL;
    MPI_Win _window = MPI_WIN_NULL;
    T* _data = nullptr;
    size_t _size = 0;

    static MPI_Comm get_intracomm(MPI_Comm comm) noexcept
    {
        MPI_Comm intracomm;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &intracomm);

        return intracomm;
    }

    static MPI_Comm get_intercomm(MPI_Comm comm, MPI_Comm intracomm) noexcept
    {
        int rank, intrarank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_rank(intracomm, &intrarank);

        MPI_Comm intercomm;
        MPI_Comm_split(comm, intrarank == 0 ? 0 : MPI_UNDEFINED, rank, &intercomm);

        return intercomm;
    }

public:
    svector(MPI_Comm comm):
        _intracomm{get_intracomm(comm)},
        _intercomm{get_intercomm(comm, _intracomm)}
    {

    }

    svector(MPI_Comm comm, size_t size):
        svector(comm)
    {
        resize(size);
    }

    svector(svector&& other) noexcept:
        _intracomm{other._intracomm},
        _intercomm{other._intercomm},
        _window{other._window},
        _data{other._data},
        _size{other._size}
    {
        other._intracomm = MPI_COMM_NULL;
        other._intercomm = MPI_COMM_NULL;
        other._window = MPI_WIN_NULL;
        other._data = nullptr;
        other._size = 0;
    }

    ~svector() noexcept
    {
        int flag;
        MPI_Finalized(&flag);
        if (flag) {
            return;
        }

        if (_window != MPI_WIN_NULL) {
            MPI_Win_free(&_window);
        }

        if (_intracomm != MPI_COMM_NULL) {
            MPI_Comm_free(&_intracomm);
        }

        if (_intercomm != MPI_COMM_NULL) {
            MPI_Comm_free(&_intercomm);
        }
    }

    inline bool root() const noexcept
    {
        return _intercomm != MPI_COMM_NULL;
    }

    inline T* data() noexcept
    {
        return _data;
    }

    inline const T* data() const noexcept
    {
        return _data;
    }

    inline size_t size() const noexcept
    {
        return _size;
    }

    void reserve(size_t size) noexcept
    {
        if (_window != MPI_WIN_NULL) {
            MPI_Win_free(&_window);
            _window = MPI_WIN_NULL;
            _data = nullptr;
        }

        MPI_Win_allocate_shared(
            root() ? sizeof(T)*size : 0,
            sizeof(T),
            MPI_INFO_NULL,
            _intracomm,
            &_data,
            &_window
        );

        MPI_Aint __size;
        int __disp_unit;
        MPI_Win_shared_query(
            _window,
            0,
            &__size,
            &__disp_unit,
            &_data
        );
    }

    void resize(size_t size) noexcept
    {
        reserve(size);
        if (root() && _data) {
            std::fill_n(_data, size, T{});
        }
        sync();

        _size = size;
    }

    inline T* begin() noexcept
    {
        return _data;
    }

    inline T* end() noexcept
    {
        return _data + _size;
    }

    inline const T* begin() const noexcept
    {
        return _data;
    }

    inline const T* end() const noexcept
    {
        return _data + _size;
    }

    inline const T* cbegin() const noexcept
    {
        return _data;
    }

    inline const T* cend() const noexcept
    {
        return _data + _size;
    }

    template<typename... Ts>
    void emplace_back(Ts&&... x) noexcept
    {
        new (_data + _size++) T(std::forward<Ts>(x)...);
    }

    template<typename I>
    inline T& operator [](I idx) noexcept
    {
        return _data[idx];
    }

    template<typename I>
    inline const T& operator [](I idx) const noexcept
    {
        return _data[idx];
    }

    inline void sync() const noexcept
    {
        if (_window == MPI_WIN_NULL) {
            return;
        }

        MPI_Win_sync(_window);
        MPI_Barrier(_intracomm);
    }

    inline void sync_all() noexcept
    {
        MPI_Bcast(&_size, 1, get_mpi_type<size_t>(), 0, _intracomm);
        sync();
    }

    template<typename U>
    inline void allreduce(MPI_Op op=MPI_SUM) const
    {
        if (root()) {
            const size_t count = sizeof(T)/sizeof(U)*_size;
            // MPI_Allreduce `count` is int; fail loudly rather than narrow silently.
            if (count > static_cast<size_t>(INT_MAX)) {
                throw std::overflow_error("svector::allreduce: MPI count exceeds INT_MAX");
            }
            MPI_Allreduce(MPI_IN_PLACE, _data, count, get_mpi_type<U>(), op, _intercomm);
        }
    }
};


#endif

