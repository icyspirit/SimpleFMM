
#ifndef MATRIX_HPP
#define MATRIX_HPP


#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

#include <ostream>
#include <stdexcept>


template<typename T>
struct is_copy_cheaper: std::conditional_t<sizeof(T) <= sizeof(void*), std::true_type, std::false_type> {};


template<typename T>
constexpr bool is_copy_cheaper_v = is_copy_cheaper<T>::value;


template<typename Element_t, typename IS, typename Derived_t>
struct Contiguous_t;


template<typename Element_t, size_t... I, typename Derived_t>
struct Contiguous_t<Element_t, std::index_sequence<I...>, Derived_t>: std::array<Element_t, sizeof...(I)> {
    using std::array<Element_t, sizeof...(I)>::at;

    constexpr Derived_t operator +(const Derived_t& __restrict other) const
    {
        return Derived_t{{(at(I) + other.at(I))...}};
    }

    constexpr Derived_t operator +(const Element_t& other) const
    {
        return Derived_t{{(at(I) + other)...}};
    }

    constexpr Derived_t operator -(const Derived_t& __restrict other) const
    {
        return Derived_t{{(at(I) - other.at(I))...}};
    }

    constexpr Derived_t operator -(const Element_t& other) const
    {
        return Derived_t{{(at(I) - other)...}};
    }

    template<typename Scalar_t, typename=std::enable_if_t<is_copy_cheaper_v<Scalar_t>>>
    constexpr Derived_t operator *(Scalar_t other) const
    {
        return Derived_t{{(at(I)*other)...}};
    }

    template<typename Scalar_t, typename=std::enable_if_t<!is_copy_cheaper_v<Scalar_t>>>
    constexpr Derived_t operator *(const Scalar_t& __restrict other) const
    {
        return Derived_t{{(at(I)*other)...}};
    }

    template<typename Scalar_t, typename=std::enable_if_t<is_copy_cheaper_v<Scalar_t>>>
    friend constexpr Derived_t operator *(Scalar_t lhs, const Derived_t& __restrict rhs)
    {
        return Derived_t{{(lhs*rhs.at(I))...}};
    }

    template<typename Scalar_t, typename=std::enable_if_t<!is_copy_cheaper_v<Scalar_t>>>
    friend constexpr Derived_t operator *(const Scalar_t& __restrict lhs, const Derived_t& __restrict rhs)
    {
        return Derived_t{{(lhs*rhs.at(I))...}};
    }

    template<typename Scalar_t, typename=std::enable_if_t<is_copy_cheaper_v<Scalar_t>>>
    constexpr Derived_t operator /(Scalar_t other) const
    {
        return Derived_t{{(at(I)/other)...}};
    }

    template<typename Scalar_t, typename=std::enable_if_t<!is_copy_cheaper_v<Scalar_t>>>
    constexpr Derived_t operator /(const Scalar_t& __restrict other) const
    {
        return Derived_t{{(at(I)/other)...}};
    }

    friend std::ostream& operator <<(std::ostream& os, const Contiguous_t& self)
    {
        if constexpr (std::is_arithmetic_v<Element_t>) {
            return ((os << self.at(I) << ' '), ...);
        } else {
            return ((os << self.at(I) << '\n'), ...);
        }
    }
};


template<typename Element_t, typename IS, typename Derived_t>
struct VectorBase;


template<typename Element_t, size_t... I, typename Derived_t>
struct VectorBase<Element_t, std::index_sequence<I...>, Derived_t>: Contiguous_t<Element_t, std::index_sequence<I...>, Derived_t> {
    using Contiguous_t<Element_t, std::index_sequence<I...>, Derived_t>::at;

    constexpr VectorBase& operator +=(const VectorBase& __restrict other)
    {
        ((at(I) += other.at(I)), ...);
        return *this;
    }

    constexpr VectorBase& operator -=(const VectorBase& __restrict other)
    {
        ((at(I) -= other.at(I)), ...);
        return *this;
    }

    template<typename Scalar_t, typename=std::enable_if_t<is_copy_cheaper_v<Scalar_t>>>
    constexpr VectorBase& operator *=(Scalar_t other)
    {
        ((at(I) *= other), ...);
        return *this;
    }

    template<typename Scalar_t, typename=std::enable_if_t<!is_copy_cheaper_v<Scalar_t>>>
    constexpr VectorBase& operator *=(const Scalar_t& __restrict other)
    {
        ((at(I) *= other), ...);
        return *this;
    }

    constexpr auto dot(const Derived_t& __restrict other) const
    {
        return ((at(I)*other.at(I)) + ...);
    }

    template<size_t M=sizeof...(I)>
    constexpr auto cross(const Derived_t& other) const
    {
        if constexpr (M != 3) {
            throw std::runtime_error("Not implemented");
        }

        return Derived_t{{at(1)*other.at(2) - at(2)*other.at(1),
                          at(2)*other.at(0) - at(0)*other.at(2),
                          at(0)*other.at(1) - at(1)*other.at(0)}};
    }

    constexpr auto norm() const
    {
        return std::sqrt(((at(I)*at(I)) + ...));
    }

    constexpr auto normalized() const
    {
        return *this/norm();
    }
};


template<typename Element_t, size_t N>
struct Vector: VectorBase<Element_t, std::make_index_sequence<N>, Vector<Element_t, N>> {};


template<typename Vector_t, typename JS, typename Derived_t>
struct MatrixBase;


template<typename Vector_t, size_t... J, typename Derived_t>
struct MatrixBase<Vector_t, std::index_sequence<J...>, Derived_t>: Contiguous_t<Vector_t, std::index_sequence<J...>, Derived_t> {
    using Contiguous_t<Vector_t, std::index_sequence<J...>, Derived_t>::at;

    constexpr auto dot(const Vector<typename Vector_t::value_type, sizeof...(J)>& __restrict other) const
    {
        return ((at(J)*other.at(J)) + ...);
    }

    constexpr auto ldot(const Vector_t& __restrict other) const
    {
        return Vector<typename Vector_t::value_type, sizeof...(J)>{{(other.dot(at(J)))...}};
    }

    template<size_t N=sizeof...(J)>
    constexpr auto determinant() const
    {
        if constexpr (N != 3) {
            throw std::runtime_error("Not implemented");
        }

        return at(0).cross(at(1)).dot(at(2));
    }

    template<size_t N=sizeof...(J)>
    constexpr auto inversed() const
    {
        if constexpr (N != 3) {
            throw std::runtime_error("Not implemented");
        }

        return Derived_t{{at(1).at(1)*at(2).at(2) - at(1).at(2)*at(2).at(1),
                          at(2).at(1)*at(0).at(2) - at(2).at(2)*at(0).at(1),
                          at(0).at(1)*at(1).at(2) - at(0).at(2)*at(1).at(1),
                          at(1).at(2)*at(2).at(0) - at(1).at(0)*at(2).at(2),
                          at(2).at(2)*at(0).at(0) - at(2).at(0)*at(0).at(2),
                          at(0).at(2)*at(1).at(0) - at(0).at(0)*at(1).at(2),
                          at(1).at(0)*at(2).at(1) - at(1).at(1)*at(2).at(0),
                          at(2).at(0)*at(0).at(1) - at(2).at(1)*at(0).at(0),
                          at(0).at(0)*at(1).at(1) - at(0).at(1)*at(1).at(0)}}/determinant();
    }
};


template<typename Element_t, size_t M, size_t N>
struct Matrix: MatrixBase<Vector<Element_t, M>, std::make_index_sequence<N>, Matrix<Element_t, M, N>> {};


#endif

