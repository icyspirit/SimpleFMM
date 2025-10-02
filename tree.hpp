
#ifndef TREE_HPP
#define TREE_HPP


#include "aligned.hpp"
#include "distributed.hpp"
#include "zorder.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#ifndef NDEBUG
#include <ostream>
#endif


inline constexpr int absi(int m) noexcept
{
    return m >= 0 ? m : -m;
}


template<typename index_t, typename zindex_t>
class children {
    static constexpr size_t dim = 3;

private:
    zindex_t _z;
    Int3<index_t, zindex_t> _ijk;

public:
    children(zindex_t z):
        _z{z << dim},
        _ijk{Int3<index_t, zindex_t>::from_z(_z)}
    {

    }

    inline children& operator ++() noexcept
    {
        ++_z;
        _ijk.i = (_ijk.i & ~1) | ((_z >> 0)%2);
        _ijk.j = (_ijk.j & ~1) | ((_z >> 1)%2);
        _ijk.k = (_ijk.k & ~1) | ((_z >> 2)%2);
        return *this;
    }

    inline bool end() const noexcept
    {
        return _z%(1 << dim) == 0;
    }

    inline zindex_t z() const noexcept
    {
        return _z;
    }

    inline const Int3<index_t, zindex_t>& ijk() const noexcept
    {
        return _ijk;
    }
};


template<typename index_t, typename zindex_t>
class neighbor {
    static constexpr size_t dim = 3;

private:
    const Int3<index_t, zindex_t> _ijkb;
    Int3<index_t, zindex_t> _dijk;
    index_t _count;

    neighbor(const Int3<index_t, zindex_t>& ijkb, const Int3<index_t, zindex_t>& ijke):
        _ijkb{ijkb},
        _dijk{ijke - ijkb},
        _count{0}
    {

    }

    neighbor(index_t i, index_t j, index_t k, index_t imax):
        neighbor({std::max(i - 1, 0), std::max(j - 1, 0), std::max(k - 1, 0)},
                 {std::min(i + 2, imax), std::min(j + 2, imax), std::min(k + 2, imax)})
    {

    }

public:
    neighbor(int l, zindex_t z):
        neighbor(z2i<dim, 0, index_t>(z), z2i<dim, 1, index_t>(z), z2i<dim, 2, index_t>(z), 1 << l)
    {

    }

    inline neighbor& operator ++() noexcept
    {
        ++_count;
        return *this;
    }

    inline bool end() const noexcept
    {
        return _count >= _dijk.i*_dijk.j*_dijk.k;
    }

    inline Int3<index_t, zindex_t> ijk() const noexcept
    {
        return {_ijkb.i + _count%_dijk.i,
                _ijkb.j + (_count/_dijk.i)%_dijk.j,
                _ijkb.k + _count/(_dijk.i*_dijk.j)};
    }

    inline zindex_t z() const noexcept
    {
        return ijk().z();
    }
};


template<typename index_t, typename zindex_t>
class interaction_list {
    static constexpr size_t dim = 3;

private:
    const Int3<index_t, zindex_t> _ijk;
    neighbor<index_t, zindex_t> _neigh_iter;
    children<index_t, zindex_t> _child_iter;

public:
    interaction_list(int l, zindex_t z):
        _ijk{Int3<index_t, zindex_t>::from_z(z)},
        _neigh_iter(l - 1, z >> dim),
        _child_iter(_neigh_iter.z())
    {
        if (is_neighbor(_ijk, _child_iter.ijk())) {
            ++*this;
        }
    }

    static inline bool is_neighbor(const Int3<index_t, zindex_t>& lhs, const Int3<index_t, zindex_t>& rhs) noexcept
    {
        return (absi(lhs.i - rhs.i) <= 1) &&
               (absi(lhs.j - rhs.j) <= 1) &&
               (absi(lhs.k - rhs.k) <= 1);
    }

    inline interaction_list& operator ++() noexcept
    {
        if ((++_child_iter).end()) {
            if ((++_neigh_iter).end()) {
                return *this;
            }
            _child_iter = children<index_t, zindex_t>(_neigh_iter.z());
        }
        return is_neighbor(_ijk, _child_iter.ijk()) ? ++*this : *this;
    }

    inline bool end() const noexcept
    {
        return _neigh_iter.end() && _child_iter.end();
    }

    inline const Int3<index_t, zindex_t>& ijk() const noexcept
    {
        return _child_iter.ijk();
    }

    inline zindex_t z() const noexcept
    {
        return _child_iter.z();
    }
};


template<typename T>
class LevelData {
private:
    svector<T> _data;

public:
    LevelData(int size, MPI_Comm shm_comm):
        _data(shm_comm, size)
    {

    }

    inline void initialize() noexcept
    {
        if (_data.root() && _data.data()) {
            std::fill_n(_data.data(), _data.size(), T{});
        }
        sync();
    }

    inline void sync() const noexcept
    {
        _data.sync();
    }

    template<typename U>
    inline void allreduce(MPI_Op op=MPI_SUM) const noexcept
    {
        _data.template allreduce<U>(op);
    }

    template<typename index_t>
    __attribute__((pure)) inline T& operator [](index_t idx) noexcept
    {
        return _data[idx];
    }

    template<typename index_t>
    __attribute__((pure)) inline const T& operator [](index_t idx) const noexcept
    {
        return _data[idx];
    }
};


#endif

