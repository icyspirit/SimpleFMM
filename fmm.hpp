
#ifndef FMM_HPP
#define FMM_HPP


#include "aligned.hpp"
#include "csr.hpp"
#include "distributed.hpp"
#include "expquad.hpp"
#include "matrix.hpp"
#include "partition.hpp"
#include "tree.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <tuple>
#include <vector>
#ifndef NDEBUG
#include <iostream>
#endif

#ifdef FMM_MEASURE_TIMING
#include "timer.hpp"
#else
#define tic(x) ((void)0)
#define toc(x) ((void)0)
#endif


template<typename T>
constexpr int get_expansion_order(T tol)
// Petersen, Smith, and Soelvason (1995)
{
    constexpr T c = 0.4;
    constexpr T a = 0.75;
    constexpr int Lmin = 8;

    int L = 0;
    T expansion = a;
    while (c*expansion/(L + 1) > tol) {
        ++L;
        expansion *= a;
    }

    return L < Lmin ? Lmin : L;
}


// The rate is the geometric one, sqrt(3)/4: a source box's half-diagonal over
// the closest centre separation an interaction list allows.  Only the constant
// is measured, and it has to be: it is the one part that depends on how the
// charges sit rather than on where the boxes are.  Relative L2 error of the far
// field against direct summation, N = 50000:
//
//   p           4        6        8       10       12       14       16       18
//   cube    6.4e-4   8.1e-5   1.4e-5   2.2e-6   6.2e-7   1.1e-7   2.0e-8   7.1e-9
//   torus        -        -        -        -   4.8e-7   1.0e-7   2.4e-8        -
//   model   4.9e-4   9.2e-5   1.7e-5   3.2e-6   6.1e-7   1.1e-7   2.1e-8   4.0e-9
//
// Fitting the rate as well gives 0.4445 over this range, so the geometry has it.
//
// The model stops holding past p = 18, where the measured curve flattens out
// around 1e-10 as the O(p^3) rotation recursions accumulate roundoff -- the
// plane-wave path, which does less arithmetic per pair, sits a factor of two
// below it there.  Asking for less than 1e-9 therefore returns an order that is
// one or two short.  This is the far field alone; a caller that contracts it
// against test functions pays its own factor on top.
template<typename T>
constexpr int get_expansion_order_empirical(T tol)
{
    constexpr T c = 0.014;
    constexpr T a = 0.4330127018922193;   // sqrt(3)/4
    constexpr int Lmin = 0;

    int L = 0;
    T expansion = 1;
    while (c*expansion > tol) {
        ++L;
        expansion *= a;
    }

    return L < Lmin ? Lmin : L;
}


enum ParticleRole : unsigned char {
    SOURCE = 1,
    TARGET = 2,
    BOTH = SOURCE | TARGET
};


inline constexpr int nm2i(int n, int m) noexcept
{
    return n*(n + 1) + m;
}


template<int p>
inline constexpr int isqrt(int n) noexcept
{
    for (int i=p; i>0; --i) {
        if (i*i <= n) {
            return i;
        }
    }
    return 0;
}


inline constexpr int nposm2i(int n, int m) noexcept
{
    return n*(n + 1)/2 + absi(m);
}


inline int pow1(int exp) noexcept
{
    return exp%2 == 0 ? 1 : -1;
}


template<typename T>
inline T Apos(int n, int m) noexcept
{
    assert(n >= absi(m));
    return 1/std::sqrt(std::tgamma(n - m + 1)*std::tgamma(n + m + 1));
}


template<typename T>
inline T Z(int n, int m, T theta) noexcept
{
    return std::sqrt(4*M_PI/(2*n + 1))*std::sph_legendre(n, absi(m), theta);
}


inline long double wigner_d(int n, int a, int b,
                            const long double* f, const long double* cbp, const long double* sbp) noexcept
{
    const long double pref = std::sqrt(f[n + a]*f[n - a]*f[n + b]*f[n - b]);
    long double sum = 0;
    for (int t=std::max(0, b - a); t<=std::min(n + b, n - a); ++t) {
        const long double term = pref/(f[n + b - t]*f[t]*f[a - b + t]*f[n - a - t])*
                                 cbp[2*n + b - a - 2*t]*sbp[a - b + 2*t];
        sum += (a - b + t)%2 == 0 ? term : -term;
    }
    return sum;
}


inline int fkm(int k, int m) noexcept
{
    return (absi(k) - absi(m) - absi(k - m)) >> 1;
}


template<typename T, size_t N, int p, template<typename, size_t> typename Container, bool support_gradient=false>
class FMM3D {
    using index_t = default_index_t;
    using zindex_t = default_zindex_t;
    static constexpr int dim = 3;
    using Coord_t = Vector<T, dim>;
    using Partitioner_t = OctreePartitioner<T, Container>;
    static constexpr int minimum_level = 2;

private:
    const Partitioner_t& _partitioner;
    const MPI_Comm _comm;
    const svector<Coord_t>& _positions;
    int _n_particle;
    T _width;

    int _size, _rank;
    // Solid harmonics are held only for the particles this rank reduces and
    // evaluates, in the order N2M and L2N walk them, so no index map is needed.
    std::vector<unsigned char> _role;
    std::vector<int> _local_off;
    std::vector<int> _src_off;
    std::vector<int> _tgt_off;
    std::vector<int> _source_index;
    std::vector<int> _target_index;
    std::vector<int> _tgt_order;
    std::vector<std::vector<int>> _leaf_load;
    int _n2n_i0 = 0;
    int _n2n_i1 = 0;
    std::vector<int> _tgt_displ;
    std::vector<int> _tgt_counts;
    std::vector<int> _tgt_displs;
    mutable std::vector<Vector<T, N>> _Qlocal;
    mutable std::vector<Vector<T, N>> _Ulocal;

    void check_mpi_count() const
    {
        // MPI counts are int; fail loudly rather than narrow silently.
        if (static_cast<std::size_t>(N)*_n_particle > static_cast<std::size_t>(INT_MAX)) {
            throw std::overflow_error("FMM3D: MPI count exceeds INT_MAX");
        }
    }
    avector<T> _phi;
    avector<std::array<T, nm2i(p, p) + 1>> _Zrho;
    avector<std::array<Vector<std::complex<T>, dim>, nm2i(p, p) + 1>> _Zgrho;

    std::array<T, nposm2i(p, p) + 1> _Z_near_positive;
    std::array<std::array<T, nposm2i(2*p, 2*p) + 1>, 64> _Z_ilist_positive;
    std::array<std::array<std::complex<T>, 4*p + 1>, 49> _expphi_ilist;
    std::array<std::array<std::complex<T>, 2*p + 1>, 4> _expphi_child;
    std::array<std::array<std::complex<T>, 2*p + 1>, 4> _expphi_parent;
    avector<T> _A;

    std::array<int, nm2i(p, p) + 1> _i2n;
    std::array<int, nm2i(p, p) + 1> _i2m;
    avector<T> _m2l_cu;
    avector<T> _m2l_cl;
    avector<int> _m2l_zi;
    avector<int> _m2l_q;
    avector<int> _m2l_pw;
    static constexpr int n_rot_class = 70;
    static constexpr int n_rot = (p + 1)*(2*p + 1)*(2*p + 3)/3;
    static constexpr std::array<int, 19> _t2i{0, 1, 2, -1, 3, 4, -1, -1, 5, 6, 7, -1, -1, 8, -1, -1, -1, -1, 9};
    std::array<int, p + 1> _rot_off;
    avector<T> _rot;
    std::array<int, 2*p + 1> _tz_off;
    avector<T> _tz;

    static constexpr int exp_digits =
#ifdef FMM_EXP_DIGITS
        FMM_EXP_DIGITS;
#else
        exp_digits_of(p);
#endif
    using EQ = ExpQuad<exp_digits>;
    static constexpr int n_exp = exp_size<exp_digits>();
    static constexpr int n_dexp = 2*7*7;
    std::array<int, EQ::s + 1> _exp_off;
    std::array<std::complex<T>, p + 1> _exp_mi;
    avector<std::complex<T>> _exp_eim;
    avector<std::complex<T>> _dexp;
    int _exp_rot_class;
    std::array<std::array<std::complex<T>, 4*p + 1>, 3> _exp_ephi;
    mutable avector<Vector<std::complex<T>, N>> _exp_w;
    mutable std::vector<int> _exp_slot;
    mutable std::vector<int> _exp_src;

    mutable std::vector<LevelData<std::array<Vector<std::complex<T>, N>, nm2i(p, p) + 1>>> _M;
    mutable std::vector<LevelData<std::array<Vector<std::complex<T>, N>, nm2i(p, p) + 1>>> _L;

    template<size_t... I>
    inline static Vector<std::complex<T>, N> r2c_impl(const Vector<T, N>& r, std::complex<T> e, std::index_sequence<I...>) noexcept
    {
        return {r[I]*e...};
    }

    inline static Vector<std::complex<T>, N> r2c(const Vector<T, N>& r, std::complex<T> e) noexcept
    {
        return r2c_impl(r, e, std::make_index_sequence<N>{});
    }

    template<size_t... I>
    inline static Vector<T, N> c2r_impl(const Vector<std::complex<T>, N>& c, std::complex<T> e, std::index_sequence<I...>) noexcept
    {
        return {(c[I].real()*e.real() - c[I].imag()*e.imag())...};
    }

    inline static Vector<T, N> c2r(const Vector<std::complex<T>, N>& c, std::complex<T> e) noexcept
    {
        return c2r_impl(c, e, std::make_index_sequence<N>{});
    }

public:
    FMM3D(const Partitioner_t& partitioner, MPI_Comm comm,
          std::vector<unsigned char> roles = {}):
        _partitioner{partitioner},
        _comm{comm},
        _positions{partitioner.positions()},
        _n_particle{static_cast<int>(_positions.size())},
        _width{partitioner.box().width()}
    {
        MPI_Comm_size(_comm, &_size);
        MPI_Comm_rank(_comm, &_rank);

        _role = std::move(roles);
        if (_role.empty()) {
            _role.assign(_n_particle, BOTH);
        } else if (static_cast<int>(_role.size()) != _n_particle) {
            throw std::runtime_error("FMM3D: one role per particle is required");
        }

        const int n_level = _partitioner.level();

        // N2M walks the sources and L2N the targets out of one leaf range, so
        // the range has to balance their sum.  With every particle in both
        // lists that weight is uniform, and the cuts have to stay where they
        // were, so weigh by the particle count instead of doubling it.
        const bool weighted = std::any_of(_role.cbegin(), _role.cend(),
                                          [](unsigned char r) noexcept { return r != BOTH; });
        _leaf_load.resize(n_level + 1);
        for (int l=0; l<=n_level; ++l) {
            const auto& olevel = _partitioner.octreeLevel(l);
            const auto& indices = olevel.indices();
            auto& scan = _leaf_load[l];
            scan.assign(olevel.n_leaf() + 1, 0);
            for (int i_leaf=0; i_leaf<olevel.n_leaf(); ++i_leaf) {
                int load = 0;
                for (int inz=0; inz<indices.nnz(i_leaf); ++inz) {
                    const unsigned char r = _role[std::get<0>(indices.value(i_leaf, inz))];
                    load += weighted ? ((r & SOURCE) != 0) + ((r & TARGET) != 0) : 1;
                }
                scan[i_leaf + 1] = scan[i_leaf] + load;
            }
        }

        {
            int n_target = 0;
            for (int i=0; i<_n_particle; ++i) {
                n_target += (_role[i] & TARGET) != 0;
            }
            const auto cut = [&](int target) noexcept {
                int seen = 0;
                for (int i=0; i<_n_particle; ++i) {
                    if (seen >= target) {
                        return i;
                    }
                    seen += (_role[i] & TARGET) != 0;
                }
                return _n_particle;
            };
            _n2n_i0 = cut(begin(n_target, _size, _rank));
            _n2n_i1 = cut(end(n_target, _size, _rank));
        }
        _local_off.resize(n_level + 2);
        _src_off.resize(n_level + 2);
        _tgt_off.resize(n_level + 2);
        _local_off[0] = 0;
        _src_off[0] = 0;
        _tgt_off[0] = 0;
        for (int l=0; l<=n_level; ++l) {
            const auto& olevel = _partitioner.octreeLevel(l);
            const auto& indices = olevel.indices();
            const auto [leaf0, leaf1] = balanced_leaf_range(l, _rank);
            int n_all = 0, n_src = 0, n_tgt = 0;
            for (int i_leaf=leaf0; i_leaf<leaf1; ++i_leaf) {
                for (int inz=0; inz<indices.nnz(i_leaf); ++inz) {
                    const unsigned char r = _role[std::get<0>(indices.value(i_leaf, inz))];
                    ++n_all;
                    n_src += (r & SOURCE) != 0;
                    n_tgt += (r & TARGET) != 0;
                }
            }
            _local_off[l + 1] = _local_off[l] + n_all;
            _src_off[l + 1] = _src_off[l] + n_src;
            _tgt_off[l + 1] = _tgt_off[l] + n_tgt;
        }
        _phi.resize(_local_off[n_level + 1]);
        _Zrho.resize(_local_off[n_level + 1]);
        _Zgrho.resize(static_cast<std::size_t>(_src_off[n_level + 1])*support_gradient);
        _source_index.reserve(_src_off[n_level + 1]);
        _target_index.reserve(_tgt_off[n_level + 1]);

        for (int l=0; l<=n_level; ++l) {
            const auto& olevel = _partitioner.octreeLevel(l);
            const auto& indices = olevel.indices();
            const auto [leaf0, leaf1] = balanced_leaf_range(l, _rank);
            int slot = _local_off[l];
            int src = _src_off[l];
            for (int i_leaf=leaf0; i_leaf<leaf1; ++i_leaf) {
                const Coord_t center = _partitioner.box().get_center(l, olevel.c_node()[olevel.from_leaf(i_leaf)]);
                for (int inz=0; inz<indices.nnz(i_leaf); ++inz, ++slot) {
                    const int i = std::get<0>(indices.value(i_leaf, inz));
                    if (_role[i] & SOURCE) {
                        _source_index.emplace_back(i);
                    }
                    if (_role[i] & TARGET) {
                        _target_index.emplace_back(i);
                    }

                    const Coord_t delta = _partitioner.box().rotate(_positions[i]) - center;
                    const T rho = delta.norm();
                    const T theta = std::acos(delta[2]/rho);
                    _phi[slot] = std::atan2(delta[1], delta[0]);
                    T rhon = 1;
                    for (int n=0; n<=p; ++n) {
                        for (int m=-n; m<=n; ++m) {
                            _Zrho[slot][nm2i(n, m)] = rhon*Z(n, m, theta);
                        }
                        rhon *= rho;
                    }

                    if constexpr (support_gradient) {
                        if (_role[i] & SOURCE) {
                            const Matrix<std::complex<T>, dim, dim> rot{
                                std::sin(theta)*std::cos(_phi[slot]), std::sin(theta)*std::sin(_phi[slot]), std::cos(theta),
                                std::cos(theta)*std::cos(_phi[slot]), std::cos(theta)*std::sin(_phi[slot]), -std::sin(theta),
                                -std::sin(_phi[slot]), std::cos(_phi[slot]), 0
                            };

                            T rhonm1 = 1;
                            for (int n=1; n<=p; ++n) {
                                for (int m=-n; m<=n; ++m) {
                                    const Vector<std::complex<T>, dim> vec{
                                        std::polar(n*rhonm1*Z(n, m, theta), -m*_phi[slot]),
                                        std::polar(rhonm1/std::sin(theta)*(std::sqrt(static_cast<T>((n + 1)*(n + 1) - m*m))*Z(n + 1, m, theta) - (n + 1)*std::cos(theta)*Z(n, m, theta)), -m*_phi[slot]),
                                        std::polar(m*rhonm1/std::sin(theta)*Z(n, m, theta), -(m*_phi[slot] + M_PI/2))
                                    };
                                    _Zgrho[src][nm2i(n, m)] = _partitioner.box().template unrotate<std::complex<T>>(rot.dot(vec));
                                }
                                rhonm1 *= rho;
                            }
                        }
                    }

                    src += (_role[i] & SOURCE) != 0;
                }
            }
        }

        for (int n=0; n<=p; ++n) {
            for (int m=0; m<=n; ++m) {
                _Z_near_positive[nposm2i(n, m)] = Z(n, m, std::acos(1/std::sqrt(static_cast<T>(dim))));
            }
        }

        for (int k=0; k<4; ++k) {
            for (int j=0; j<4; ++j) {
                for (int i=0; i<4; ++i) {
                    if (i + j + k >= 2) {
                        for (int n=0; n<=2*p; ++n) {
                            for (int m=0; m<=n; ++m) {
                                _Z_ilist_positive[16*k + 4*j + i][nposm2i(n, m)] =
                                    Z(n, m, std::acos(k/std::sqrt(static_cast<T>(i*i + j*j + k*k))));
                            }
                        }
                    }
                }
            }
        }

        for (int j=-3; j<=3; ++j) {
            for (int i=-3; i<=3; ++i) {
                const T phi = std::atan2(static_cast<T>(j), static_cast<T>(i));
                for (int q=-2*p; q<=2*p; ++q) {
                    _expphi_ilist[7*(j + 3) + (i + 3)][q + 2*p] = std::polar(static_cast<T>(1), q*phi);
                }
            }
        }

        for (int c=0; c<4; ++c) {
            for (int q=-p; q<=p; ++q) {
                _expphi_child[c][q + p] = std::polar(static_cast<T>(1), q*get_phi_of_child(c));
                _expphi_parent[c][q + p] = std::polar(static_cast<T>(1), q*get_phi_of_parent(c));
            }
        }

        _A.resize(nposm2i(2*p, 2*p) + 1);
        for (int n=0; n<=2*p; ++n) {
            for (int m=0; m<=n; ++m) {
                _A[nposm2i(n, m)] = Apos<T>(n, m);
            }
        }

        int i = 0;
        for (int n=0; n<=p; ++n) {
            for (int m=-n; m<=n; ++m) {
                _i2n[i] = n;
                _i2m[i] = m;
                ++i;
            }
        }

        constexpr std::size_t n_m2l_term = static_cast<std::size_t>(nm2i(p, p) + 1)*(nm2i(p, p) + 1);
        _m2l_cu.resize(n_m2l_term);
        _m2l_cl.resize(n_m2l_term);
        _m2l_zi.resize(n_m2l_term);
        _m2l_q.resize(n_m2l_term);
        _m2l_pw.resize(n_m2l_term);
        for (int jk=0; jk<=nm2i(p, p); ++jk) {
            const int j = _i2n[jk];
            const int k = _i2m[jk];
            for (int nm=0; nm<=nm2i(p, p); ++nm) {
                const int n = _i2n[nm];
                const int m = _i2m[nm];
                const std::size_t idx = static_cast<std::size_t>(jk)*(nm2i(p, p) + 1) + nm;
                _m2l_cu[idx] = pow1(fkm(k - m, k) + n)*
                               _A[nposm2i(n, m)]*_A[nposm2i(j, k)]/_A[nposm2i(j + n, m - k)];
                _m2l_cl[idx] = _m2l_cu[idx]*pow1((j + n) - (m - k));
                _m2l_zi[idx] = nposm2i(j + n, m - k);
                _m2l_q[idx] = (m - k) + 2*p;
                _m2l_pw[idx] = j + n;
            }
        }

        _rot_off[0] = 0;
        for (int n=1; n<=p; ++n) {
            _rot_off[n] = _rot_off[n - 1] + (2*n - 1)*(2*n - 1);
        }

        const auto sgn = [](int m) noexcept { return m < 0 ? pow1(m) : 1; };

        long double f[2*p + 2];
        f[0] = 1;
        for (int n=1; n<2*p + 2; ++n) {
            f[n] = f[n - 1]*n;
        }

        _rot.resize(static_cast<std::size_t>(n_rot_class)*n_rot);
        constexpr std::array<int, 10> tvals{0, 1, 2, 4, 5, 8, 9, 10, 13, 18};
        for (int ti=0; ti<10; ++ti) {
            for (int dk=-3; dk<=3; ++dk) {
                if (tvals[ti] == 0 && dk == 0) {
                    continue;
                }
                const long double theta = std::acos(dk/std::sqrt(static_cast<long double>(tvals[ti] + dk*dk)));
                long double cbp[2*p + 1], sbp[2*p + 1];
                cbp[0] = 1;
                sbp[0] = 1;
                for (int e=0; e<2*p; ++e) {
                    cbp[e + 1] = cbp[e]*std::cos(-theta/2);
                    sbp[e + 1] = sbp[e]*std::sin(-theta/2);
                }
                T* R = _rot.data() + static_cast<std::size_t>(7*ti + dk + 3)*n_rot;
                for (int n=0; n<=p; ++n) {
                    for (int a=-n; a<=n; ++a) {
                        for (int b=-n; b<=n; ++b) {
                            R[_rot_off[n] + (a + n)*(2*n + 1) + (b + n)] =
                                sgn(a)*sgn(b)*wigner_d(n, a, b, f, cbp, sbp);
                        }
                    }
                }
            }
        }

        {
            int off = 0;
            for (int k=-p; k<=p; ++k) {
                _tz_off[k + p] = off;
                off += (p + 1 - absi(k))*(p + 1 - absi(k));
            }
            _tz.resize(off);
            for (int k=-p; k<=p; ++k) {
                const int ak = absi(k);
                const int w = p + 1 - ak;
                for (int j=ak; j<=p; ++j) {
                    for (int n=ak; n<=p; ++n) {
                        _tz[_tz_off[k + p] + (j - ak)*w + (n - ak)] =
                            pow1(n + ak)*_A[nposm2i(n, ak)]*_A[nposm2i(j, ak)]/_A[nposm2i(j + n, 0)];
                    }
                }
            }
        }

        const auto fill_owner_order = [&](unsigned char role, std::vector<int>& order, std::vector<int>& displ) {
            order.reserve(_n_particle);
            displ.assign(_size + 1, 0);
            for (int r=0; r<_size; ++r) {
                for (int l=0; l<=n_level; ++l) {
                    const auto& olevel = _partitioner.octreeLevel(l);
                    const auto& indices = olevel.indices();
                    const auto [leaf0, leaf1] = balanced_leaf_range(l, r);
                    for (int i_leaf=leaf0; i_leaf<leaf1; ++i_leaf) {
                        for (int inz=0; inz<indices.nnz(i_leaf); ++inz) {
                            const int i = std::get<0>(indices.value(i_leaf, inz));
                            if (_role[i] & role) {
                                order.emplace_back(i);
                            }
                        }
                    }
                }
                displ[r + 1] = static_cast<int>(order.size());
            }
        };
        fill_owner_order(TARGET, _tgt_order, _tgt_displ);

        _tgt_counts.resize(_size);
        _tgt_displs.resize(_size);
        for (int r=0; r<_size; ++r) {
            _tgt_counts[r] = N*(_tgt_displ[r + 1] - _tgt_displ[r]);
            _tgt_displs[r] = N*_tgt_displ[r];
        }
        _Qlocal.resize(_source_index.size());
        _Ulocal.resize(_target_index.size());

        {
            _exp_off[0] = 0;
            for (int k=0; k<EQ::s; ++k) {
                _exp_off[k + 1] = _exp_off[k] + EQ::M[k]/2;
            }

            for (int t=0; t<=p; ++t) {
                _exp_mi[t] = std::polar(static_cast<T>(1), static_cast<T>(t)*static_cast<T>(M_PI/2));
            }

            _exp_eim.resize(static_cast<std::size_t>(n_exp)*(2*p + 1));
            for (int k=0; k<EQ::s; ++k) {
                for (int j=0; j<EQ::M[k]/2; ++j) {
                    const T alpha = 2*M_PI*j/EQ::M[k];
                    std::complex<T>* eim = _exp_eim.data() + static_cast<std::size_t>(_exp_off[k] + j)*(2*p + 1);
                    for (int m=-p; m<=p; ++m) {
                        eim[m + p] = std::polar(static_cast<T>(1), m*alpha);
                    }
                }
            }

            _dexp.resize(static_cast<std::size_t>(n_dexp)*n_exp);
            for (int tz=2; tz<=3; ++tz) {
                for (int ty=-3; ty<=3; ++ty) {
                    for (int tx=-3; tx<=3; ++tx) {
                        std::complex<T>* D = _dexp.data() + static_cast<std::size_t>(dexp_index(tx, ty, tz))*n_exp;
                        for (int k=0; k<EQ::s; ++k) {
                            const T lam = EQ::lambda[k];
                            const T decay = std::exp(-lam*tz);
                            for (int j=0; j<EQ::M[k]/2; ++j) {
                                const T alpha = 2*M_PI*j/EQ::M[k];
                                D[_exp_off[k] + j] = decay*std::polar(static_cast<T>(1),
                                                                      lam*(tx*std::cos(alpha) + ty*std::sin(alpha)));
                            }
                        }
                    }
                }
            }

            const Int3<index_t, zindex_t> ex{1, 0, 0};
            const Int3<index_t, zindex_t> ey{0, 1, 0};
            _exp_rot_class = rot_class(ex);
            for (int q=0; q<=4*p; ++q) {
                _exp_ephi[0][q] = 1;
                _exp_ephi[1][q] = get_expphi_of_other(ey)[q];
                _exp_ephi[2][q] = get_expphi_of_other(ex)[q];
            }
        }

        _M.reserve(_partitioner.level() + 1);
        _L.reserve(_partitioner.level() + 1);
        for (int l=0; l<=_partitioner.level(); ++l) {
            _M.emplace_back(_partitioner.octreeLevel(l).n_node(), _comm);
            _L.emplace_back(_partitioner.octreeLevel(l).n_node(), _comm);
        }

        if (get_rank() == 0) {
            std::cout << "FMM with p = " << p << ", n_particle = " << _n_particle << std::endl;
        }
    }

    const auto& partitioner() const noexcept
    {
        return _partitioner;
    }

    inline const std::vector<int>& target_owner_order() const noexcept
    {
        return _tgt_order;
    }

    inline const std::vector<int>& target_owner_displ() const noexcept
    {
        return _tgt_displ;
    }

    inline int n_source() const noexcept
    {
        return static_cast<int>(_source_index.size());
    }

    inline int n_target() const noexcept
    {
        return static_cast<int>(_target_index.size());
    }

    // Global indices of the particles this rank reduces, and of the ones it
    // evaluates, in the order N2M and L2N walk them.  Each belongs to exactly
    // one rank.  Without roles the two lists are the same.
    inline const std::vector<int>& source_index() const noexcept
    {
        return _source_index;
    }

    inline const std::vector<int>& target_index() const noexcept
    {
        return _target_index;
    }

    inline MPI_Comm comm() const noexcept
    {
        return _comm;
    }

    inline T get_phi_of_child(zindex_t c) const noexcept
    {
        if (c%4 == 0) {
            return -0.75*M_PI;
        } else if (c%4 == 1) {
            return -0.25*M_PI;
        } else if (c%4 == 2) {
            return 0.75*M_PI;
        } else {
            return 0.25*M_PI;
        }
    }

    inline T get_Z_of_child(zindex_t c, int n, int m) const noexcept
    {
        return (((c >> 2)%2 == 0) ? pow1(n - absi(m)) : 1)*_Z_near_positive[nposm2i(n, m)];
    }

    inline T get_phi_of_parent(zindex_t c) const noexcept
    {
        if (c%4 == 0) {
            return 0.25*M_PI;
        } else if (c%4 == 1) {
            return 0.75*M_PI;
        } else if (c%4 == 2) {
            return -0.25*M_PI;
        } else {
            return -0.75*M_PI;
        }
    }

    inline T get_Z_of_parent(zindex_t c, int n, int m) const noexcept
    {
        return (((c >> 2)%2 == 0) ? 1 : pow1(n - absi(m)))*_Z_near_positive[nposm2i(n, m)];
    }

    template<typename Int3_t>
    inline auto& get_Z_of_other(const Int3_t& dijk) const noexcept
    {
        return _Z_ilist_positive[16*absi(dijk.k) + 4*absi(dijk.j) + absi(dijk.i)];
    }

    template<typename Int3_t>
    inline auto& get_expphi_of_other(const Int3_t& dijk) const noexcept
    {
        return _expphi_ilist[7*(dijk.j + 3) + (dijk.i + 3)];
    }

    template<typename Int3_t>
    inline int rot_class(const Int3_t& dijk) const noexcept
    {
        return 7*_t2i[dijk.i*dijk.i + dijk.j*dijk.j] + (dijk.k + 3);
    }

    inline T get_A(int n, int m) const noexcept
    {
        return _A[nposm2i(n, m)];
    }

    // Split [0, n_leaf) so ranks get similar loads rather than leaf counts.
    std::pair<int, int> balanced_leaf_range(int l, int rank) const noexcept
    {
        const auto& scan = _leaf_load[l];
        const int n_leaf = static_cast<int>(scan.size()) - 1;
        const long long total = scan[n_leaf];
        const auto cut = [&](long long target) noexcept {
            int lo = 0;
            int hi = n_leaf;
            while (lo < hi) {
                const int mid = (lo + hi)/2;
                if (scan[mid] < target) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return lo;
        };

        return {cut(begin(total, _size, rank)), cut(end(total, _size, rank))};
    }

    template<bool gradient=false, typename LevelData_t>
    void N2M(int l, const Vector<T, N>* Q, LevelData_t& M) const noexcept
    {
        static_assert(support_gradient || !gradient);

        const auto& olevel = _partitioner.octreeLevel(l);
        const auto& indices = olevel.indices();

        const auto [leaf0, leaf1] = balanced_leaf_range(l, _rank);
        int slot = _local_off[l];
        int src = _src_off[l];
        for (int i_leaf=leaf0; i_leaf<leaf1; ++i_leaf) {
            const int i_node = olevel.from_leaf(i_leaf);
            for (int inz=0; inz<indices.nnz(i_leaf); ++inz, ++slot) {
                if (!(_role[std::get<0>(indices.value(i_leaf, inz))] & SOURCE)) {
                    continue;
                }
                std::array<std::complex<T>, p + 1> E;
                if constexpr (!gradient) {
                    const std::complex<T> estep{std::cos(_phi[slot]), -std::sin(_phi[slot])};
                    E[0] = 1;
                    for (int m=1; m<=p; ++m) {
                        E[m] = E[m - 1]*estep;
                    }
                }
                for (int nm=0; nm<=nm2i(p, p); ++nm) {
                    if constexpr (!gradient) {
                        const int m = _i2m[nm];
                        M[i_node][nm] += r2c(Q[src]*_Zrho[slot][nm], m >= 0 ? E[m] : std::conj(E[-m]));
                    } else {
                        static_assert(N == 3);
                        M[i_node][nm] += Vector<std::complex<T>, N>{Q[src][0], Q[src][1], Q[src][2]}.cross(_Zgrho[src][nm]);
                    }
                }
                ++src;
            }
        }
    }

    template<typename ClusterData_t>
    void M2Mc(int lc, zindex_t cc, const ClusterData_t& Mc, ClusterData_t& Mp) const noexcept
    {
        const T rho = std::sqrt(static_cast<T>(0.75))*_width/(1 << lc);
        const std::complex<T>* __restrict ephi = _expphi_child[cc%4].data();

        for (int j=0; j<=p; ++j) {
            for (int k=-j; k<=j; ++k) {
                T rhopn = 1;
                for (int n=0; n<=j; ++n) {
                    for (int m=std::max(-n, k - (j - n)); m<=std::min(n, k + (j - n)); ++m) {
                        const T v = pow1(fkm(k, m))*get_A(n, m)*get_A(j - n, k - m)/get_A(j, k)*rhopn*get_Z_of_child(cc, n, -m);
                        Mp[nm2i(j, k)] += v*ephi[p - m]*Mc[nm2i(j - n, k - m)];
                    }
                    rhopn *= rho;
                }
            }
        }
    }

    template<typename LevelData_t>
    void M2M(int lp, const LevelData_t& Mc, LevelData_t& Mp) const noexcept
    {
        const auto& olevel = _partitioner.octreeLevel(lp);
        const auto& clist = olevel.clist();

        for (int ip=begin(olevel.n_node(), _size, _rank); ip<end(olevel.n_node(), _size, _rank); ++ip) {
            for (int inz=0; inz<clist.nnz(ip); ++inz) {
                const auto& [ic, cc] = clist.value(ip, inz);
                M2Mc(lp + 1, cc, Mc[ic], Mp[ip]);
            }
        }
    }

    template<typename Int3_t, typename ClusterData_t>
    void M2Lc_direct(int l, const Int3_t& dijk, const ClusterData_t& Ml, ClusterData_t& Ll) const noexcept
    {
        const T rho = _width/(1 << l)*std::sqrt(static_cast<T>(dijk.i*dijk.i + dijk.j*dijk.j + dijk.k*dijk.k));
        const std::complex<T>* __restrict expphi = get_expphi_of_other(dijk).data();
        const T* __restrict Zi = get_Z_of_other(dijk).data();
        const T* __restrict coef = (dijk.k >= 0 ? _m2l_cu : _m2l_cl).data();
        const int* __restrict zi = _m2l_zi.data();
        const int* __restrict qi = _m2l_q.data();
        const int* __restrict pw = _m2l_pw.data();

        alignas(64) T powrho[2*p + 1];
        powrho[0] = 1/rho;
        for (int i=0; i<2*p; ++i) {
            powrho[i + 1] = powrho[i]/rho;
        }

        for (int jk=0; jk<=nm2i(p, p); ++jk) {
            const std::size_t base = static_cast<std::size_t>(jk)*(nm2i(p, p) + 1);
            Vector<std::complex<T>, N> acc{};
            for (int nm=0; nm<=nm2i(p, p); ++nm) {
                const T v = coef[base + nm]*powrho[pw[base + nm]]*Zi[zi[base + nm]];
                acc += v*expphi[qi[base + nm]]*Ml[nm];
            }
            Ll[jk] += acc;
        }
    }

    template<typename Int3_t, typename ClusterData_t>
    void M2Lc_rtr(int l, const Int3_t& dijk, const ClusterData_t& Ml, ClusterData_t& Ll) const noexcept
    {
        using CV = Vector<std::complex<T>, N>;

        const T rho = _width/(1 << l)*std::sqrt(static_cast<T>(dijk.i*dijk.i + dijk.j*dijk.j + dijk.k*dijk.k));
        const std::complex<T>* __restrict expphi = get_expphi_of_other(dijk).data();
        const T* __restrict R = _rot.data() + static_cast<std::size_t>(rot_class(dijk))*n_rot;

        alignas(64) T powrho[2*p + 1];
        powrho[0] = 1/rho;
        for (int i=0; i<2*p; ++i) {
            powrho[i + 1] = powrho[i]/rho;
        }

        std::array<CV, nm2i(p, p) + 1> Mt;
        for (int nm=0; nm<=nm2i(p, p); ++nm) {
            Mt[nm] = expphi[_i2m[nm] + 2*p]*Ml[nm];
        }

        std::array<CV, nm2i(p, p) + 1> W;
        for (int n=0; n<=p; ++n) {
            const T* __restrict Rn = R + _rot_off[n];
            const int base = nm2i(n, -n);
            for (int a=0; a<2*n + 1; ++a) {
                const T* __restrict row = Rn + a*(2*n + 1);
                CV acc{};
                for (int b=0; b<2*n + 1; ++b) {
                    acc += row[b]*Mt[base + b];
                }
                W[base + a] = acc;
            }
        }

        std::array<CV, nm2i(p, p) + 1> Lz;
        for (int k=-p; k<=p; ++k) {
            const int ak = absi(k);
            const int w = p + 1 - ak;
            const T* __restrict tz = _tz.data() + _tz_off[k + p];
            for (int j=ak; j<=p; ++j) {
                const T* __restrict row = tz + (j - ak)*w;
                CV acc{};
                for (int n=ak; n<=p; ++n) {
                    acc += (row[n - ak]*powrho[j + n])*W[nm2i(n, k)];
                }
                Lz[nm2i(j, k)] = acc;
            }
        }

        for (int j=0; j<=p; ++j) {
            const T* __restrict Rj = R + _rot_off[j];
            const int base = nm2i(j, -j);
            std::array<CV, 2*p + 1> Lt;
            for (int a=0; a<2*j + 1; ++a) {
                Lt[a] = CV{};
            }
            for (int b=0; b<2*j + 1; ++b) {
                const T* __restrict row = Rj + b*(2*j + 1);
                const CV v = Lz[base + b];
                #pragma omp simd
                for (int a=0; a<2*j + 1; ++a) {
                    Lt[a] += row[a]*v;
                }
            }
            for (int a=0; a<2*j + 1; ++a) {
                Ll[base + a] += expphi[(j - a) + 2*p]*Lt[a];
            }
        }
    }

    template<typename Int3_t, typename ClusterData_t>
    void M2Lc(int l, const Int3_t& dijk, const ClusterData_t& Ml, ClusterData_t& Ll) const noexcept
    {
#ifdef FMM_M2L_DIRECT
        M2Lc_direct(l, dijk, Ml, Ll);
#else
        M2Lc_rtr(l, dijk, Ml, Ll);
#endif
    }

    static constexpr int dexp_index(int tx, int ty, int tz) noexcept
    {
        return (tz - 2)*49 + (ty + 3)*7 + (tx + 3);
    }

    template<typename Int3_t>
    static bool exp_direction(const Int3_t& dijk, int dir, int& tx, int& ty, int& tz) noexcept
    {
        const int i = -dijk.i;
        const int j = -dijk.j;
        const int k = -dijk.k;

        switch (dir) {
        case 0: if (k >= 2)  { tx =  i; ty =  j; tz =  k; return true; } break;
        case 1: if (k <= -2) { tx =  i; ty =  j; tz = -k; return true; } break;
        case 2: if (absi(k) <= 1 && j >= 2)  { tx = -k; ty = -i; tz =  j; return true; } break;
        case 3: if (absi(k) <= 1 && j <= -2) { tx = -k; ty = -i; tz = -j; return true; } break;
        case 4: if (absi(k) <= 1 && absi(j) <= 1 && i >= 2)  { tx = -k; ty = j; tz =  i; return true; } break;
        case 5: if (absi(k) <= 1 && absi(j) <= 1 && i <= -2) { tx = -k; ty = j; tz = -i; return true; } break;
        }

        return false;
    }

    template<typename ClusterData_t>
    void exp_reflect(ClusterData_t& c) const noexcept
    {
        for (int nm=0; nm<=nm2i(p, p); ++nm) {
            if ((_i2n[nm] + absi(_i2m[nm]))%2) {
                c[nm] = static_cast<T>(-1)*c[nm];
            }
        }
    }

    template<typename ClusterData_t>
    void exp_rotate(int axis, const ClusterData_t& Ml, ClusterData_t& out) const noexcept
    {
        using CV = Vector<std::complex<T>, N>;

        if (axis == 0) {
            for (int nm=0; nm<=nm2i(p, p); ++nm) {
                out[nm] = Ml[nm];
            }
            return;
        }

        const std::complex<T>* __restrict ephi = _exp_ephi[axis].data();
        const T* __restrict R = _rot.data() + static_cast<std::size_t>(_exp_rot_class)*n_rot;

        std::array<CV, nm2i(p, p) + 1> Mt;
        for (int nm=0; nm<=nm2i(p, p); ++nm) {
            Mt[nm] = ephi[_i2m[nm] + 2*p]*Ml[nm];
        }

        for (int n=0; n<=p; ++n) {
            const T* __restrict Rn = R + _rot_off[n];
            const int base = nm2i(n, -n);
            for (int a=0; a<2*n + 1; ++a) {
                const T* __restrict row = Rn + a*(2*n + 1);
                CV acc{};
                for (int b=0; b<2*n + 1; ++b) {
                    acc += row[b]*Mt[base + b];
                }
                out[base + a] = acc;
            }
        }
    }

    template<typename ClusterData_t, typename OutData_t>
    void exp_unrotate(int axis, const ClusterData_t& Lz, OutData_t& Ll) const noexcept
    {
        using CV = Vector<std::complex<T>, N>;

        if (axis == 0) {
            for (int nm=0; nm<=nm2i(p, p); ++nm) {
                Ll[nm] += Lz[nm];
            }
            return;
        }

        const std::complex<T>* __restrict ephi = _exp_ephi[axis].data();
        const T* __restrict R = _rot.data() + static_cast<std::size_t>(_exp_rot_class)*n_rot;

        for (int j=0; j<=p; ++j) {
            const T* __restrict Rj = R + _rot_off[j];
            const int base = nm2i(j, -j);
            std::array<CV, 2*p + 1> Lt;
            for (int a=0; a<2*j + 1; ++a) {
                Lt[a] = CV{};
            }
            for (int b=0; b<2*j + 1; ++b) {
                const T* __restrict row = Rj + b*(2*j + 1);
                const CV v = Lz[base + b];
                for (int a=0; a<2*j + 1; ++a) {
                    Lt[a] += row[a]*v;
                }
            }
            for (int a=0; a<2*j + 1; ++a) {
                Ll[base + a] += ephi[(j - a) + 2*p]*Lt[a];
            }
        }
    }

    template<size_t... I>
    inline static Vector<std::complex<T>, N> vconj_impl(const Vector<std::complex<T>, N>& v, std::index_sequence<I...>) noexcept
    {
        return {std::conj(v[I])...};
    }

    inline static Vector<std::complex<T>, N> vconj(const Vector<std::complex<T>, N>& v) noexcept
    {
        return vconj_impl(v, std::make_index_sequence<N>{});
    }

    template<typename ClusterData_t>
    void M2X(T d, const ClusterData_t& M, Vector<std::complex<T>, N>* __restrict W) const noexcept
    {
        using CV = Vector<std::complex<T>, N>;

        for (int k=0; k<EQ::s; ++k) {
            const T lam = static_cast<T>(EQ::lambda[k])/d;

            std::array<CV, p + 1> F;
            for (int m=0; m<=p; ++m) {
                T pw = std::pow(lam, m);
                CV f{};
                for (int n=m; n<=p; ++n) {
                    f += (_A[nposm2i(n, m)]*pw)*M[nm2i(n, m)];
                    pw *= lam;
                }
                F[m] = f;
            }

            const T pref = static_cast<T>(EQ::w[k])/d/EQ::M[k];
            for (int j=0; j<EQ::M[k]/2; ++j) {
                const int e = _exp_off[k] + j;
                const std::complex<T>* __restrict eim = _exp_eim.data() + static_cast<std::size_t>(e)*(2*p + 1);
                CV acc = F[0];
                for (int m=1; m<=p; ++m) {
                    const CV z = eim[m + p]*F[m];
                    acc += _exp_mi[m]*(z + vconj(z));
                }
                W[e] = pref*acc;
            }
        }
    }

    template<typename ClusterData_t>
    void X2L(T d, const Vector<std::complex<T>, N>* __restrict V, ClusterData_t& L) const noexcept
    {
        using CV = Vector<std::complex<T>, N>;

        for (int k=0; k<EQ::s; ++k) {
            const T lam = static_cast<T>(EQ::lambda[k])/d;

            std::array<CV, p + 1> G;
            for (int m=0; m<=p; ++m) {
                G[m] = CV{};
            }
            for (int j=0; j<EQ::M[k]/2; ++j) {
                const int e = _exp_off[k] + j;
                const std::complex<T>* __restrict eim = _exp_eim.data() + static_cast<std::size_t>(e)*(2*p + 1);
                const CV c = vconj(V[e]);
                const CV even = V[e] + c;
                const CV odd = V[e] - c;
                for (int m=0; m<=p; ++m) {
                    G[m] += eim[p - m]*(m%2 ? odd : even);
                }
            }

            for (int m=0; m<=p; ++m) {
                const CV g = _exp_mi[m]*G[m];
                const CV gc = vconj(g);
                T pw = std::pow(-lam, m);
                for (int n=m; n<=p; ++n) {
                    const T a = _A[nposm2i(n, m)]*pw;
                    L[nm2i(n, m)] += a*g;
                    if (m > 0) {
                        L[nm2i(n, -m)] += a*gc;
                    }
                    pw *= -lam;
                }
            }
        }
    }

    template<typename LevelData_t>
    void M2L_exp(int l, const LevelData_t& Ml, LevelData_t& Ll) const noexcept
    {
        using CV = Vector<std::complex<T>, N>;
        using Cluster_t = std::array<CV, nm2i(p, p) + 1>;

        const auto& olevel = _partitioner.octreeLevel(l);
        const auto& ilist = olevel.ilist();
        const T d = _width/(1 << l);
        const int nn = olevel.n_node();
        const int i0 = begin(nn, _size, _rank);
        const int i1 = end(nn, _size, _rank);

        _exp_slot.assign(nn, -1);
        _exp_src.clear();

        std::vector<CV> V(n_exp);
        Cluster_t Mr, Lr;

        for (int dir=0; dir<6; ++dir) {
            const int axis = dir/2;
            const bool refl = dir%2;

            for (const int b: _exp_src) {
                _exp_slot[b] = -1;
            }
            _exp_src.clear();
            for (int i=i0; i<i1; ++i) {
                for (int inz=0; inz<ilist.nnz(i); ++inz) {
                    const auto& [ii, dijk] = ilist.value(i, inz);
                    int tx, ty, tz;
                    if (exp_direction(dijk, dir, tx, ty, tz) && _exp_slot[ii] < 0) {
                        _exp_slot[ii] = static_cast<int>(_exp_src.size());
                        _exp_src.push_back(ii);
                    }
                }
            }
            if (_exp_src.empty()) {
                continue;
            }

            _exp_w.resize(_exp_src.size()*n_exp);
            for (std::size_t si=0; si<_exp_src.size(); ++si) {
                exp_rotate(axis, Ml[_exp_src[si]], Mr);
                if (refl) {
                    exp_reflect(Mr);
                }
                M2X(d, Mr, _exp_w.data() + si*n_exp);
            }

            for (int i=i0; i<i1; ++i) {
                bool any = false;
                for (int inz=0; inz<ilist.nnz(i); ++inz) {
                    const auto& [ii, dijk] = ilist.value(i, inz);
                    int tx, ty, tz;
                    if (!exp_direction(dijk, dir, tx, ty, tz)) {
                        continue;
                    }
                    if (!any) {
                        std::fill(V.begin(), V.end(), CV{});
                        any = true;
                    }
                    const std::complex<T>* __restrict D =
                        _dexp.data() + static_cast<std::size_t>(dexp_index(tx, ty, tz))*n_exp;
                    const CV* __restrict W = _exp_w.data() + static_cast<std::size_t>(_exp_slot[ii])*n_exp;
                    for (int e=0; e<n_exp; ++e) {
                        V[e] += D[e]*W[e];
                    }
                }
                if (!any) {
                    continue;
                }

                for (int nm=0; nm<=nm2i(p, p); ++nm) {
                    Lr[nm] = CV{};
                }
                X2L(d, V.data(), Lr);
                if (refl) {
                    exp_reflect(Lr);
                }
                exp_unrotate(axis, Lr, Ll[i]);
            }
        }
    }

    template<typename LevelData_t>
    void M2L(int l, const LevelData_t& Ml, LevelData_t& Ll) const noexcept
    {
        const auto& olevel = _partitioner.octreeLevel(l);
        const auto& ilist = olevel.ilist();

        for (int i=begin(olevel.n_node(), _size, _rank); i<end(olevel.n_node(), _size, _rank); ++i) {
            for (int inz=0; inz<ilist.nnz(i); ++inz) {
                const auto& [ii, dijk] = ilist.value(i, inz);
                M2Lc(l, dijk, Ml[ii], Ll[i]);
            }
        }
    }

    template<typename ClusterData_t>
    void L2Lc(int lc, zindex_t cc, const ClusterData_t& Lp, ClusterData_t& Lc) const noexcept
    {
        const T rho = std::sqrt(static_cast<T>(0.75))*_width/(1 << lc);
        const std::complex<T>* __restrict ephi = _expphi_parent[cc%4].data();

        for (int j=0; j<=p; ++j) {
            for (int k=-j; k<=j; ++k) {
                T rhomjpn = 1;
                for (int n=j; n<=p; ++n) {
                    for (int m=std::max(-n, k - (n - j)); m<=std::min(n, k + (n - j)); ++m) {
                        const T v = pow1(fkm(m, m - k) + n + j)*get_A(n - j, m - k)*get_A(j, k)/get_A(n, m)*
                                    rhomjpn*get_Z_of_parent(cc, n - j, m - k);
                        Lc[nm2i(j, k)] += v*ephi[(m - k) + p]*Lp[nm2i(n, m)];
                    }
                    rhomjpn *= rho;
                }
            }
        }
    }

    template<typename LevelData_t>
    void L2L(int lc, const LevelData_t& Lp, LevelData_t& Lc) const noexcept
    {
        const auto& olevel = _partitioner.octreeLevel(lc);
        const auto& plist = olevel.plist();

        for (int ic=begin(olevel.n_node(), _size, _rank); ic<end(olevel.n_node(), _size, _rank); ++ic) {
            const auto& [ip, cc] = plist[ic];
            L2Lc(lc, cc, Lp[ip], Lc[ic]);
        }
    }

    template<typename LevelData_t>
    void L2N(int l, const LevelData_t& L, Vector<T, N>* __restrict U) const noexcept
    {
        const auto& olevel = _partitioner.octreeLevel(l);
        const auto& indices = olevel.indices();

        const auto [leaf0, leaf1] = balanced_leaf_range(l, _rank);
        int slot = _local_off[l];
        int tgt = _tgt_off[l];
        for (int i_leaf=leaf0; i_leaf<leaf1; ++i_leaf) {
            const int i_node = olevel.from_leaf(i_leaf);
            for (int inz=0; inz<indices.nnz(i_leaf); ++inz, ++slot) {
                if (!(_role[std::get<0>(indices.value(i_leaf, inz))] & TARGET)) {
                    continue;
                }
                std::array<std::complex<T>, p + 1> F;
                const std::complex<T> estep{std::cos(_phi[slot]), std::sin(_phi[slot])};
                F[0] = 1;
                for (int m=1; m<=p; ++m) {
                    F[m] = F[m - 1]*estep;
                }
                #pragma omp simd
                for (int jk=0; jk<=nm2i(p, p); ++jk) {
                    const int m = _i2m[jk];
                    U[tgt] += _Zrho[slot][jk]*c2r(L[i_node][jk], m >= 0 ? F[m] : std::conj(F[-m]));
                }
                ++tgt;
            }
        }
    }

    template<bool gradient=false, typename IsSelf>
    void N2N(const Vector<T, N>* Q, Vector<T, N>* U, const IsSelf& is_self) const noexcept
    {
        static_assert(support_gradient || !gradient);

        for (int i=_n2n_i0; i<_n2n_i1; ++i) {
            if (!(_role[i] & TARGET)) {
                continue;
            }
            const auto& [li, i_leaf] = _partitioner.get_partition(i);
            const auto& nlist = _partitioner.octreeLevel(li).nlist();
            for (int inz=0; inz<nlist.nnz(i_leaf); ++inz) {
                const auto& [lj, j_leaf] = nlist.value(i_leaf, inz);
                const auto& indices = _partitioner.octreeLevel(lj).indices();
                for (int jnz=0; jnz<indices.nnz(j_leaf); ++jnz) {
                    const int j = std::get<0>(indices.value(j_leaf, jnz));
                    if ((_role[j] & SOURCE) && !is_self(i, j)) {
                        if constexpr (!gradient) {
                            U[i] += Q[j]/(_positions[i] - _positions[j]).norm();
                        } else {
                            static_assert(N == 3);
                            const auto R = _positions[i] - _positions[j];
                            U[i] += Q[j].cross(R)/std::pow(R.norm(), static_cast<T>(3));
                        }
                    }
                }
            }

        }
    }

    // L2N accumulates, so the caller's buffer starts clean.
    template<bool gradient=false>
    void far_field_local(const Vector<T, N>* Q, Vector<T, N>* U) const
    {
        static_assert(support_gradient || !gradient);

        std::fill_n(U, _target_index.size(), Vector<T, N>{});

        const int level = _partitioner.level();

        if (level < minimum_level) {
            return;
        }

        for (auto& M: _M) {
            M.initialize();
        }
        for (auto& L: _L) {
            L.initialize();
        }

        tic("N2M2M");
        N2M<gradient>(level, Q, _M[level]);
        for (int l=level - 1; l>=minimum_level; --l) {
            _M[l + 1].sync();
            _M[l + 1].template allreduce<T>();
            _M[l + 1].sync();
            M2M(l, _M[l + 1], _M[l]);
            N2M<gradient>(l, Q, _M[l]);
        }
        _M[minimum_level].sync();
        _M[minimum_level].template allreduce<T>();
        _M[minimum_level].sync();
        toc("N2M2M");

        tic("M2L");
        for (int l=minimum_level; l<=level; ++l) {
#ifdef FMM_M2L_ROTATION
            M2L(l, _M[l], _L[l]);
#else
            M2L_exp(l, _M[l], _L[l]);
#endif
        }
        toc("M2L");

        tic("L2L2N");
        for (int l=minimum_level; l<level; ++l) {
            _L[l].sync();
            _L[l].template allreduce<T>();
            _L[l].sync();
            L2N(l, _L[l], U);
            L2L(l + 1, _L[l], _L[l + 1]);
        }
        _L[level].sync();
        _L[level].template allreduce<T>();
        _L[level].sync();
        L2N(level, _L[level], U);
        toc("L2L2N");
    }

    // Q and U hold one entry per particle.  Each rank supplies whatever share
    // of the source term it has and the sum over the communicator is what is
    // applied; U comes back complete on every rank.  Q is workspace: it is not
    // preserved, which is what keeps this from needing a buffer of its own.
    template<bool gradient=false>
    void rinv_far(Vector<T, N>* Q, Vector<T, N>* U) const
    {
        static_assert(support_gradient || !gradient);
        check_mpi_count();

        // A reduce-scatter moves half the bytes but measures ~2x slower here:
        // the irregular form misses the tuned paths an allreduce lands on.
        MPI_Allreduce(MPI_IN_PLACE, &Q[0][0], N*_n_particle, get_mpi_type<T>(), MPI_SUM, _comm);
        for (std::size_t k=0; k<_source_index.size(); ++k) {
            _Qlocal[k] = Q[_source_index[k]];
        }

        far_field_local<gradient>(_Qlocal.data(), _Ulocal.data());

        // Each result has exactly one producer, so gathering beats reducing.
        MPI_Allgatherv(&_Ulocal[0][0], N*static_cast<int>(_target_index.size()), get_mpi_type<T>(),
                       &Q[0][0], _tgt_counts.data(), _tgt_displs.data(), get_mpi_type<T>(), _comm);
        // Without roles every particle is a target and the scatter covers U.
        if (_tgt_order.size() != static_cast<std::size_t>(_n_particle)) {
            std::fill_n(U, _n_particle, Vector<T, N>{});
        }
        for (std::size_t k=0; k<_tgt_order.size(); ++k) {
            U[_tgt_order[k]] = Q[k];
        }
    }

    template<bool gradient=false, typename IsSelf>
    void rinv(Vector<T, N>* Q, Vector<T, N>* U, const IsSelf& is_self) const
    {
        static_assert(support_gradient || !gradient);
        check_mpi_count();

        // The near field reads Q at arbitrary neighbours, so unlike the far
        // field this one cannot be given each rank only its own share.  Q is
        // completed in place rather than consumed, and stays that way.
        MPI_Allreduce(MPI_IN_PLACE, &Q[0][0], N*_n_particle, get_mpi_type<T>(), MPI_SUM, _comm);

        for (std::size_t k=0; k<_source_index.size(); ++k) {
            _Qlocal[k] = Q[_source_index[k]];
        }
        far_field_local<gradient>(_Qlocal.data(), _Ulocal.data());

        std::fill_n(U, _n_particle, Vector<T, N>{});
        for (std::size_t k=0; k<_target_index.size(); ++k) {
            U[_target_index[k]] = _Ulocal[k];
        }

        tic("N2N");
        N2N<gradient>(Q, U, is_self);
        toc("N2N");

        // N2N and L2N divide the particles differently, so an entry of U can
        // have been written by two ranks at once.
        MPI_Allreduce(MPI_IN_PLACE, &U[0][0], N*_n_particle, get_mpi_type<T>(), MPI_SUM, _comm);
    }

    template<bool gradient=false>
    void rinv(Vector<T, N>* Q, Vector<T, N>* U) const
    {
        static_assert(support_gradient || !gradient);

        rinv<gradient>(Q, U, [](int i, int j) { return i == j; });
    }
};


#ifndef FMM_MEASURE_TIMING
#undef tic
#undef toc
#endif


#endif

