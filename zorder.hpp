
#ifndef ZORDER_HPP
#define ZORDER_HPP


#include <cassert>
#include <cstddef>
#ifndef NDEBUG
#include <ostream>
#endif
#include <utility>
#include <vector>


#define BITS_PER_BYTE (8)

using default_index_t = int32_t;
using default_zindex_t = int64_t;


template<typename index_t, typename zindex_t, size_t dim,
         typename=std::enable_if_t<std::conjunction_v<std::is_signed<index_t>, std::is_signed<zindex_t>>>>
inline constexpr size_t bits_per_index = std::min(BITS_PER_BYTE*sizeof(index_t), (BITS_PER_BYTE*sizeof(zindex_t) - 1)/dim);


template<size_t dim, size_t IJK, typename zindex_t, size_t... I, typename index_t, typename=std::enable_if_t<IJK < dim>>
inline zindex_t i2z_impl(index_t i, std::index_sequence<I...>) noexcept
{
    return ((static_cast<zindex_t>(i & (static_cast<index_t>(1) << I)) << ((dim - 1)*I + IJK)) | ...);
}


template<size_t dim, size_t IJK, typename zindex_t, typename index_t, typename=std::enable_if_t<IJK < dim>>
inline zindex_t i2z(index_t i) noexcept
{
    return i2z_impl<dim, IJK, zindex_t>(i, std::make_index_sequence<bits_per_index<index_t, zindex_t, dim> - 1>{});
}


template<typename zindex_t, typename index_t>
inline zindex_t ijk2z(index_t i, index_t j, index_t k) noexcept
{
    return i2z<3, 0, zindex_t>(i) | i2z<3, 1, zindex_t>(j) | i2z<3, 2, zindex_t>(k);
}


template<size_t dim, size_t IJK, typename index_t, size_t... I, typename zindex_t, typename=std::enable_if_t<IJK < dim>>
inline index_t z2i_impl(zindex_t z, std::index_sequence<I...>) noexcept
{
    return (((z & (static_cast<zindex_t>(1) << (dim*I + IJK))) >> ((dim - 1)*I + IJK)) | ...);
}


template<size_t dim, size_t IJK, typename index_t, typename zindex_t, typename=std::enable_if_t<IJK < dim>>
inline index_t z2i(zindex_t z) noexcept
{
    return z2i_impl<dim, IJK, index_t>(z, std::make_index_sequence<bits_per_index<index_t, zindex_t, dim> - 1>{});
}


template<typename index_t, typename zindex_t>
struct Int3 {
    static constexpr size_t dim = 3;

    index_t i: bits_per_index<index_t, zindex_t, dim>;
    index_t j: bits_per_index<index_t, zindex_t, dim>;
    index_t k: bits_per_index<index_t, zindex_t, dim>;

    static Int3 from_z(zindex_t z) noexcept
    {
        return {z2i<dim, 0, index_t>(z), z2i<dim, 1, index_t>(z), z2i<dim, 2, index_t>(z)};
    }

    inline zindex_t z() const noexcept
    {
        return ijk2z<zindex_t>(i, j, k);
    }
 
    friend inline Int3 operator -(const Int3& lhs, const Int3& rhs) noexcept
    {
        return {lhs.i - rhs.i, lhs.j - rhs.j, lhs.k - rhs.k};
    }

    friend inline bool operator <(const Int3& lhs, const Int3& rhs) noexcept
    {
        if (lhs.k != rhs.k) {
            return lhs.k < rhs.k;
        }
        if (lhs.j != rhs.j) {
            return lhs.j < rhs.j;
        }
        return lhs.i < rhs.i;
    }

#ifndef NDEBUG
    friend std::ostream& operator <<(std::ostream& os, const Int3& self)
    {
        return os << '(' << self.i << ", " << self.j << ", " << self.k << ')';
    }
#endif
};


#endif

