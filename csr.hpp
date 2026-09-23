
#ifndef CSR_HPP
#define CSR_HPP


#include "distributed.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#ifndef NDEBUG
#include <ostream>
#endif


template<typename... Ts>
class CSRP {
private:
    int _nrow;
    std::vector<int> _rowPtr;
    std::vector<std::tuple<int, Ts...>> _values;

public:
    CSRP(int nrow, const std::vector<int>& rowPtr, const std::vector<std::tuple<int, Ts...>>& values):
        _nrow{nrow}, _rowPtr{rowPtr}, _values{values}
    {

    }

    CSRP():
        _nrow{0}
    {
        _rowPtr.emplace_back(0);
    }

    auto& operator ++() noexcept
    {
        if (_nrow > 0) {
            std::sort(_values.begin() + _rowPtr[_nrow - 1], _values.begin() + _rowPtr[_nrow]);
                      //[](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });
        }
        _rowPtr.emplace_back(_rowPtr[_nrow++]);
        return *this;
    }

    void finish() noexcept
    {
        if (_nrow > 0) {
            std::sort(_values.begin() + _rowPtr[_nrow - 1], _values.begin() + _rowPtr[_nrow]);
                      //[](const auto& lhs, const auto& rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });
        }
        _rowPtr.shrink_to_fit();
        _values.shrink_to_fit();
    }

    inline int nrow() const noexcept
    {
        return _nrow;
    }

    inline int nnz(int begin, int end) const noexcept
    {
        return _rowPtr[end] - _rowPtr[begin];
    }

    inline int nnz(int row) const noexcept
    {
        return nnz(row, row + 1);
    }

    inline int nnz() const noexcept
    {
        return nnz(0, _nrow);
    }

    inline void reserve_nrow(int nrow) noexcept
    {
        _rowPtr.reserve(nrow + 1);
    }

    inline void reserve(int nnz) noexcept
    {
        _values.reserve(nnz);
    }

    inline void emplace_back(int col, Ts... value) noexcept
    {
        _rowPtr[_nrow]++;
        _values.emplace_back(col, value...);
    }

    inline bool binary_search(int row, const std::tuple<int, Ts...>& value) const noexcept
    {
        return std::binary_search(_values.cbegin() + _rowPtr[row], _values.cbegin() + _rowPtr[row + 1], value);
    }

    __attribute__((pure)) inline const std::tuple<int, Ts...>& value(int row, int n) const noexcept
    {
        assert(_rowPtr[row] + n < _rowPtr[row + 1]);
        return _values[_rowPtr[row] + n];
    }

    CSRP<> get_inverse_mapping(int nrow) const noexcept
    {
        std::vector<int> rowPtr(nrow + 1);
        for (int i=0; i<_nrow; ++i) {
            for (int j=_rowPtr[i]; j<_rowPtr[i + 1]; ++j) {
                rowPtr[std::get<0>(_values[j]) + 1]++;
            }
        }

        for (int i=0; i<nrow; ++i) {
            rowPtr[i + 1] += rowPtr[i];
        }

        std::vector<std::tuple<int>> values(rowPtr[nrow]);
        auto rowPtr_ = rowPtr;
        for (int i=0; i<_nrow; ++i) {
            for (int j=_rowPtr[i]; j<_rowPtr[i + 1]; ++j) {
                values[rowPtr_[std::get<0>(_values[j])]++] = i;
            }
        }

        return CSRP<>(nrow, rowPtr, values);
    }

#ifndef NDEBUG
    friend std::ostream& operator <<(std::ostream& os, const CSRP& self)
    {
        os << "Nrow = " << self._nrow << std::endl;
        for (int i=0; i<self._nrow; ++i) {
            os << "Row = " << i << ", nnz = " << self.nnz(i) << std::endl;
            os << "Cols = ";
            for (int j=self._rowPtr[i]; j<self._rowPtr[i + 1]; ++j) {
                os << std::get<0>(self._values[j]) << ", ";
                //os << "(" << std::get<0>(self._values[j]) << ", " << std::get<1>(self._values[j]) << "), ";
            }
            os << std::endl;
        }

        return os;
    }
#endif
};


template<typename... Ts>
class SCSRP {
public:
    using Value_t = std::tuple<int, Ts...>;

private:
    int _nrow = 0;
    svector<int> _rowPtr;
    svector<Value_t> _values;

public:
    SCSRP():
        _rowPtr(get_shm_comm()),
        _values(get_shm_comm())
    {

    }

    inline bool root() const noexcept
    {
        return _rowPtr.root();
    }

    // Collective; rowPtr/values need to be filled on the shm root only.
    void assign(const std::vector<int>& rowPtr, const std::vector<Value_t>& values)
    {
        _rowPtr.reserve(rowPtr.size());
        _values.reserve(values.size());
        if (root()) {
            for (const int p: rowPtr) {
                _rowPtr.emplace_back(p);
            }
            for (const auto& v: values) {
                _values.emplace_back(v);
            }
        }
        _rowPtr.sync_all();
        _values.sync_all();
        _nrow = static_cast<int>(_rowPtr.size()) - 1;
    }

    void reserve(int nrow, int nnz)
    {
        _rowPtr.reserve(nrow + 1);
        _values.reserve(nnz);
        if (root()) {
            _rowPtr.emplace_back(0);
        }
    }

    auto& operator ++() noexcept
    {
        const int n = static_cast<int>(_rowPtr.size()) - 1;
        if (n > 0) {
            std::sort(_values.begin() + _rowPtr[n - 1], _values.begin() + _rowPtr[n]);
        }
        _rowPtr.emplace_back(_rowPtr[n]);
        return *this;
    }

    inline void emplace_back(int col, Ts... value) noexcept
    {
        _rowPtr[static_cast<int>(_rowPtr.size()) - 1]++;
        _values.emplace_back(col, value...);
    }

    void finish() noexcept
    {
        if (root()) {
            const int n = static_cast<int>(_rowPtr.size()) - 1;
            if (n > 0) {
                std::sort(_values.begin() + _rowPtr[n - 1], _values.begin() + _rowPtr[n]);
            }
        }
        _rowPtr.sync_all();
        _values.sync_all();
        _nrow = static_cast<int>(_rowPtr.size()) - 1;
    }

    void free() noexcept
    {
        _rowPtr.free();
        _values.free();
        _nrow = 0;
    }

    inline int nrow() const noexcept
    {
        return _nrow;
    }

    inline int nnz(int begin, int end) const noexcept
    {
        return _rowPtr[end] - _rowPtr[begin];
    }

    inline int nnz(int row) const noexcept
    {
        return nnz(row, row + 1);
    }

    inline int nnz() const noexcept
    {
        return _rowPtr.size() > 0 ? nnz(0, _nrow) : 0;
    }

    inline bool binary_search(int row, const Value_t& value) const noexcept
    {
        return std::binary_search(_values.cbegin() + _rowPtr[row], _values.cbegin() + _rowPtr[row + 1], value);
    }

    __attribute__((pure)) inline const Value_t& value(int row, int n) const noexcept
    {
        return _values[_rowPtr[row] + n];
    }

    __attribute__((pure)) inline const Value_t& at(int row, int col) const noexcept
    {
        assert(row >= 0 && row < _nrow);

        int lo = _rowPtr[row], hi = _rowPtr[row + 1];
        while (lo < hi) {
            const int mid = lo + (hi - lo)/2;
            if (std::get<0>(_values[mid]) < col) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        assert(lo < _rowPtr[row + 1] && std::get<0>(_values[lo]) == col);

        return _values[lo];
    }

    SCSRP<> get_inverse_mapping(int nrow) const
    {
        std::vector<int> rowPtr;
        std::vector<std::tuple<int>> values;
        if (root()) {
            rowPtr.assign(nrow + 1, 0);
            for (int i=0; i<_nrow; ++i) {
                for (int j=_rowPtr[i]; j<_rowPtr[i + 1]; ++j) {
                    rowPtr[std::get<0>(_values[j]) + 1]++;
                }
            }

            for (int i=0; i<nrow; ++i) {
                rowPtr[i + 1] += rowPtr[i];
            }

            values.resize(rowPtr[nrow]);
            auto rowPtr_ = rowPtr;
            for (int i=0; i<_nrow; ++i) {
                for (int j=_rowPtr[i]; j<_rowPtr[i + 1]; ++j) {
                    values[rowPtr_[std::get<0>(_values[j])]++] = i;
                }
            }
        }

        SCSRP<> inverse;
        inverse.assign(rowPtr, values);
        return inverse;
    }
};

#endif
