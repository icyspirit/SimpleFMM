
#ifndef FMM_HPP
#define FMM_HPP


#include "aligned.hpp"
#include "csr.hpp"
#include "distributed.hpp"
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
#include <functional>
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


// Measured relative L2 error of the far field against direct summation, on a
// uniform cube and on a torus shell alike:
//
//   p           6        8       10       12       14       16
//   error   1.2e-4   1.9e-5   2.2e-6   6.2e-7   1.1e-7   2.9e-8
//
// which is c*a^p with a = 0.45 and c = 0.01.  Contracting the field against
// test functions costs a further factor of about 1.8, so c is set to bound
// that rather than the field itself.
template<typename T>
constexpr int get_expansion_order_empirical(T tol)
{
    constexpr T c = 0.02;
    constexpr T a = 0.45;
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
    const std::vector<Coord_t>& _positions;
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
    CSRP<> _slist;

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
          const std::function<std::vector<int>(int)>& self_generator,
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
        _local_off.resize(n_level + 2);
        _src_off.resize(n_level + 2);
        _tgt_off.resize(n_level + 2);
        _local_off[0] = 0;
        _src_off[0] = 0;
        _tgt_off[0] = 0;
        for (int l=0; l<=n_level; ++l) {
            const auto& olevel = _partitioner.octreeLevel(l);
            const auto& indices = olevel.indices();
            const auto [leaf0, leaf1] = balanced_leaf_range(indices, olevel.n_leaf(), _size, _rank);
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
            const auto [leaf0, leaf1] = balanced_leaf_range(indices, olevel.n_leaf(), _size, _rank);
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

        // Who owns what follows from the tree and the rank count alone, so each
        // rank works out every rank's share here rather than asking for it.
        const auto fill_owner_order = [&](unsigned char role, std::vector<int>& order, std::vector<int>& displ) {
            order.reserve(_n_particle);
            displ.assign(_size + 1, 0);
            for (int r=0; r<_size; ++r) {
                for (int l=0; l<=n_level; ++l) {
                    const auto& olevel = _partitioner.octreeLevel(l);
                    const auto& indices = olevel.indices();
                    const auto [leaf0, leaf1] = balanced_leaf_range(indices, olevel.n_leaf(), _size, r);
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

        _slist.reserve_nrow(_n_particle);
        _slist.reserve(_n_particle*self_generator(0).size());
        for (int i=0; i<_n_particle; ++i) {
            ++_slist;
            for (int j: self_generator(i)) {
                if (!_partitioner.is_neighbor(i, j)) {
                    _slist.emplace_back(j);
                }
            }
        }
        _slist.finish();

        _M.reserve(_partitioner.level() + 1);
        _L.reserve(_partitioner.level() + 1);
        for (int l=0; l<=_partitioner.level(); ++l) {
            _M.emplace_back(_partitioner.octreeLevel(l).n_node(), _comm);
            _L.emplace_back(_partitioner.octreeLevel(l).n_node(), _comm);
        }

        if (get_rank() == 0) {
            std::cout << "FMM with p = " << p << ", n_particle = " << _n_particle << std::endl;
            std::cout << "There are " << _slist.nnz() << " self interactions which are far enough" << std::endl;
        }
    }

    FMM3D(const Partitioner_t& partitioner, MPI_Comm comm):
        FMM3D(partitioner, comm, [](int i) { return std::vector{i}; })
    {

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

    const auto& slist() const noexcept
    {
        return _slist;
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

    // Split [0, n_leaf) so ranks get similar particle counts rather than leaf counts.
    static std::pair<int, int> balanced_leaf_range(const CSRP<>& indices, int n_leaf, int size, int rank) noexcept
    {
        const long long total = indices.nnz();
        const auto cut = [&](long long target) noexcept {
            int lo = 0;
            int hi = n_leaf;
            while (lo < hi) {
                const int mid = (lo + hi)/2;
                if (indices.nnz(0, mid) < target) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            return lo;
        };

        return {cut(begin(total, size, rank)), cut(end(total, size, rank))};
    }

    template<bool gradient=false, typename LevelData_t>
    void N2M(int l, const Vector<T, N>* Q, LevelData_t& M) const noexcept
    {
        static_assert(support_gradient || !gradient);

        const auto& olevel = _partitioner.octreeLevel(l);
        const auto& indices = olevel.indices();

        const auto [leaf0, leaf1] = balanced_leaf_range(indices, olevel.n_leaf(), _size, _rank);
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

        const auto [leaf0, leaf1] = balanced_leaf_range(indices, olevel.n_leaf(), _size, _rank);
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

        for (int i=begin(_n_particle, _size, _rank); i<end(_n_particle, _size, _rank); ++i) {
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

            for (int inz=0; inz<_slist.nnz(i); ++inz) {
                const int j = std::get<0>(_slist.value(i, inz));
                if (!(_role[j] & SOURCE)) {
                    continue;
                }
                if constexpr (!gradient) {
                    U[i] -= Q[j]/(_positions[i] - _positions[j]).norm();
                } else {
                    static_assert(N == 3);
                    const auto R = _positions[i] - _positions[j];
                    U[i] -= Q[j].cross(R)/std::pow(R.norm(), static_cast<T>(3));
                }
            }
        }
    }

    // L2N accumulates, so the caller's buffer starts clean.
    template<bool gradient=false>
    void far_field_local(const Vector<T, N>* Q, Vector<T, N>* U) const noexcept
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
            M2L(l, _M[l], _L[l]);
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
    void rinv_nonear(Vector<T, N>* Q, Vector<T, N>* U) const
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

