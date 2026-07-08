
#ifndef BOX_HPP
#define BOX_HPP


#include "zorder.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <numeric>
#include <ostream>
#include <tuple>
#include <vector>


template<typename T, int dim, template<typename, size_t> typename Container>
class Box {
public:
    using Coord_t = Container<T, dim>;
    using index_t = default_index_t;
    using zindex_t = default_zindex_t;

private:
    static constexpr T MARGIN = 1e-3;
    static constexpr int MAX_LEVEL = bits_per_index<index_t, zindex_t, dim> - 1;

    const int _X, _Y, _Z;
    const T _phi;
    T _width;
    Coord_t _corner;

    static std::pair<Coord_t, Coord_t> get_minmax(const std::vector<Coord_t>& vs) noexcept
    {
        Coord_t min{vs.front()};
        Coord_t max{vs.front()};
        for (const auto& v: vs) {
            for (int d=0; d<dim; ++d) {
                min[d] = min[d] > v[d] ? v[d] : min[d];
                max[d] = max[d] < v[d] ? v[d] : max[d];
            }
        }

        return {min, max};
    }

    static T max_element(const Coord_t& v) noexcept
    {
        T ret = v[0];
        for (int i=1; i<dim; ++i) {
            ret = ret > v[i] ? ret : v[i];
        }

        return ret;
    }

    template<size_t... I>
    inline static zindex_t c_impl(int level, const Coord_t& v, std::index_sequence<I...>) noexcept
    {
        assert(((v[I] >= 0 && v[I] <= 1) && ...));
        const index_t imax = (static_cast<index_t>(1) << level) - 1;
        return (i2z<sizeof...(I), I, zindex_t>(std::clamp(static_cast<index_t>(v[I]*(1 << level)), static_cast<index_t>(0), imax)) | ...);
    }

    template<size_t... I>
    inline static Coord_t center_impl(int level, zindex_t c, std::index_sequence<I...>) noexcept
    {
        assert(c < (static_cast<zindex_t>(1) << dim*level));
        return {((z2i<dim, I, index_t>(c) + static_cast<T>(0.5))/(1 << level))...};
    }

public:
    Box(const Coord_t& corner, T width, int vertical_axis=dim - 1, T phideg=0):
        _X{(vertical_axis + 1)%dim},
        _Y{(vertical_axis + 2)%dim},
        _Z{vertical_axis},
        _phi{phideg*static_cast<T>(M_PI/180)},
        _width{width},
        _corner{corner}
    {
        assert(vertical_axis >= 0 && vertical_axis < dim);
    }

    Box(const std::vector<Coord_t>& positions, int vertical_axis=dim - 1, T phideg=0):
        _X{(vertical_axis + 1)%dim},
        _Y{(vertical_axis + 2)%dim},
        _Z{vertical_axis},
        _phi{phideg*static_cast<T>(M_PI/180)}
    {
        assert(vertical_axis >= 0 && vertical_axis < dim);

        std::vector<Coord_t> positions_;
        positions_.reserve(positions.size());
        std::transform(positions.cbegin(), positions.cend(), std::back_inserter(positions_), [&](auto& v) { return rotate(v); });

        const auto minmax = get_minmax(positions_);
        const auto dl = minmax.second - minmax.first;
        const T dmax = max_element(dl);

        auto center = (minmax.second + minmax.first)/2;
        _width = (1 + 2*MARGIN)*dl[dim - 1];
        while (_width < dmax) {
            center[dim - 1] += _width/2;
            _width *= 2;
        }
        _corner = center - _width/2;
    }

    inline T width() const noexcept
    {
        return _width;
    }

    inline const Coord_t& corner() const noexcept
    {
        return _corner;
    }

    inline zindex_t get_c(int level, const Coord_t& v) const noexcept
    {
        return c_impl(level, (rotate(v) - _corner)/_width, std::make_index_sequence<dim>{});
    }

    auto get_cs(int level, const std::vector<Coord_t>& vs) const noexcept
    {
        std::vector<zindex_t> cs;
        cs.reserve(vs.size());
        for (const Coord_t& v: vs) {
            cs.emplace_back(get_c(level, v));
        }

        return cs;
    }

    auto get_cis(int level, const std::vector<int> indices, const std::vector<Coord_t>& vs) const noexcept
    {
        std::vector<std::pair<zindex_t, int>> cis;
        cis.reserve(indices.size());
        for (const int idx: indices) {
            cis.push_back({get_c(level, vs[idx]), idx});
        }
        std::sort(cis.begin(), cis.end());

        return cis;
    }

    inline Coord_t get_center(int level, zindex_t c) const noexcept
    {
        return _corner + _width*center_impl(level, c, std::make_index_sequence<dim>{});
    }

    inline Coord_t rotate(const Coord_t& v) const noexcept
    {
        return {std::cos(_phi)*v[_X] - std::sin(_phi)*v[_Y],
                std::sin(_phi)*v[_X] + std::cos(_phi)*v[_Y],
                v[_Z]};
    }

    inline bool comp(const Coord_t& lhs, const Coord_t& rhs) const noexcept
    {
        for (int level=1; level<=MAX_LEVEL; ++level) {
            const zindex_t clhs = get_c(level, lhs);
            const zindex_t crhs = get_c(level, rhs);
            if (clhs < crhs) {
                return true;
            }
            if (clhs > crhs) {
                return false;
            }
        }
        return false;
    }

    friend std::ostream& operator <<(std::ostream& os, const Box& self)
    {
        return os << "Box width = " << self._width << ", corner = " << self._corner;
    }
};


#endif

