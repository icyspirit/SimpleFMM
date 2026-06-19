
#ifndef PARTITION_HPP
#define PARTITION_HPP


#include "box.hpp"
#include "csr.hpp"
#include "tree.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#ifndef NDEBUG
#include <ostream>
#endif


#include <iostream>


inline std::vector<int> range(int begin, int end) noexcept
{
    std::vector<int> ret(end - begin);
    std::iota(ret.begin(), ret.end(), begin);

    return ret;
}


template<typename T, template<typename, size_t> typename Container>
class OctreeNode {
    static constexpr int dim = 3;
    using Coord_t = Container<T, dim>;
    using index_t = default_index_t;
    using zindex_t = default_zindex_t;

private:
    int _l;
    zindex_t _c;
    std::vector<int> _indices;
    std::vector<OctreeNode> _children;

    template<typename Box_t>
    void subdivide(const std::vector<Coord_t>& positions, const Box_t& box) noexcept
    {
        const auto cis = box.get_cis(_l + 1, _indices, positions);
        std::vector<zindex_t> cs(cis.size());
        std::transform(cis.cbegin(), cis.cend(), cs.begin(), [](const auto& ci) { return ci.first; });
        std::transform(cis.cbegin(), cis.cend(), _indices.begin(), [](const auto& ci) { return ci.second; });
        for (zindex_t c=(_c << dim); c<(_c << dim) + (static_cast<zindex_t>(1) << dim); ++c) {
            const std::vector<int> idx(_indices.cbegin() + std::distance(cs.cbegin(), std::lower_bound(cs.cbegin(), cs.cend(), c)),
                                       _indices.cbegin() + std::distance(cs.cbegin(), std::upper_bound(cs.cbegin(), cs.cend(), c)));
            if (!idx.empty()) {
                _children.emplace_back(_l + 1, c, idx);
            }
        }
        _indices.clear();
    }

public:
    OctreeNode(int l, zindex_t c, const std::vector<int>& indices):
        _l{l},
        _c{c},
        _indices{indices}
    {

    }

    inline int l() const noexcept
    {
        return _l;
    }

    inline zindex_t c() const noexcept
    {
        return _c;
    }

    inline const auto& indices() const noexcept
    {
        return _indices;
    }

    inline const auto& children() const noexcept
    {
        return _children;
    }

    template<typename Box_t>
    void refine(const std::vector<Coord_t>& positions, const Box_t& box, int max_level, int max_particles_per_node) noexcept
    {
        traverse(
            [&](OctreeNode* node) {
                node->subdivide(positions, box);
            },
            [&](const OctreeNode* node) {
                return node->l() < max_level && static_cast<int>(node->indices().size()) > max_particles_per_node;
            }
        );
    }

    template<typename F>
    void traverse(const F& f) const noexcept
    {
        f(this);
        for (const auto& child: _children) {
            child.traverse(f);
        }
    }

    template<typename F>
    void traverse(const F& f) noexcept
    {
        f(this);
        for (auto& child: _children) {
            child.traverse(f);
        }
    }

    template<typename F, typename G>
    void traverse(const F& f, const G& g) const noexcept
    {
        if (g(this)) {
            f(this);
            for (const auto& child: _children) {
                child.traverse(f, g);
            }
        }
    }

    template<typename F, typename G>
    void traverse(const F& f, const G& g) noexcept
    {
        if (g(this)) {
            f(this);
            for (auto& child: _children) {
                child.traverse(f, g);
            }
        }
    }
};


class OctreeLevel {
    static constexpr size_t dim = 3;
    static constexpr size_t n_ilist_max = 6*6*6 - 3*3*3;
    static constexpr size_t n_child_max = 1 << dim;
    using index_t = default_index_t;
    using zindex_t = default_zindex_t;

private:
    const int _level;
    int _n_node;
    int _n_leaf;
    std::vector<zindex_t> _c_node;
    std::vector<int> _i_leaf;
    CSRP<> _indices;

    CSRP<Int3<index_t, zindex_t>> _ilist;
    CSRP<zindex_t> _clist;
    std::vector<std::pair<int, zindex_t>> _plist;
    CSRP<int> _nlist;

public:
    template<typename Node_t>
    OctreeLevel(int level, const Node_t* root):
        _level{level}
    {
        _n_node = 0;
        _n_leaf = 0;
        size_t n_indice = 0;
        root->traverse(
            [&](const Node_t* node) {
                if (node->l() == _level) {
                    ++_n_node;
                    if (!node->indices().empty()) {
                        ++_n_leaf;
                        n_indice += node->indices().size();
                    }
                }
            },
            [&](const Node_t* node) {
                return node->l() <= _level;
            }
        );

        _c_node.reserve(_n_node);
        _i_leaf.reserve(_n_leaf);
        _indices.reserve_nrow(_n_leaf);
        _indices.reserve(n_indice);
        root->traverse(
            [&](const Node_t* node) {
                if (node->l() == _level) {
                    _c_node.emplace_back(node->c());
                    if (!node->indices().empty()) {
                        _i_leaf.emplace_back(_c_node.size() - 1);
                        ++_indices;
                        for (const int index: node->indices()) {
                            _indices.emplace_back(index);
                        }
                    }
                }
            },
            [&](const Node_t* node) {
                return node->l() <= _level;
            }
        );
    }

    inline int level() const noexcept
    {
        return _level;
    }

    inline int n_node() const noexcept
    {
        return _n_node;
    }

    inline int n_leaf() const noexcept
    {
        return _n_leaf;
    }

    inline const auto& c_node() const noexcept
    {
        return _c_node;
    }

    inline const auto& indices() const noexcept
    {
        return _indices;
    }

    inline const auto& ilist() const noexcept
    {
        return _ilist;
    }

    inline const auto& clist() const noexcept
    {
        return _clist;
    }

    inline const auto& plist() const noexcept
    {
        return _plist;
    }

    inline const auto& nlist() const noexcept
    {
        return _nlist;
    }

    inline bool has(zindex_t c) const noexcept
    {
        return std::binary_search(_c_node.cbegin(), _c_node.cend(), c);
    }

    inline int index(zindex_t c) const noexcept
    {
        return std::distance(_c_node.cbegin(), std::lower_bound(_c_node.cbegin(), _c_node.cend(), c));
    }

    inline bool is_leaf(int i) const noexcept
    {
        return std::binary_search(_i_leaf.cbegin(), _i_leaf.cend(), i);
    }

    inline int from_leaf(int i) const noexcept
    {
        return _i_leaf[i];
    }

    inline int to_leaf(int i) const noexcept
    {
        return std::distance(_i_leaf.cbegin(), std::lower_bound(_i_leaf.cbegin(), _i_leaf.cend(), i));
    }

    void create_ilist() noexcept
    {
        _ilist.reserve_nrow(_n_node);
        _ilist.reserve(_n_node*n_ilist_max);
        for (const zindex_t c: _c_node) {
            ++_ilist;
            interaction_list<index_t, zindex_t> il(_level, c);
            do {
                if (has(il.z())) {
                    _ilist.emplace_back(index(il.z()), il.ijk() - Int3<index_t, zindex_t>::from_z(c));
                }
            } while (!(++il).end());
        }
        _ilist.finish();
    }

    void create_clist(const OctreeLevel& child) noexcept
    {
        _clist.reserve_nrow(_n_node);
        _clist.reserve(_n_node*n_child_max);
        for (const zindex_t c: _c_node) {
            ++_clist;
            children<index_t, zindex_t> ch(c);
            do {
                if (child.has(ch.z())) {
                    _clist.emplace_back(child.index(ch.z()), ch.z());
                }
            } while (!(++ch).end());
        }
        _clist.finish();
    }

    void create_plist(const OctreeLevel& parent) noexcept
    {
        _plist.reserve(_n_node);
        for (const zindex_t c: _c_node) {
            assert(parent.has(c >> dim));
            _plist.push_back({parent.index(c >> dim), c});
        }
    }

    void add_nlist_ancestor(int i, const OctreeLevel& ancestor) noexcept
    {
        if (ancestor.is_leaf(i)) {
            _nlist.emplace_back(ancestor.level(), ancestor.to_leaf(i));
        }
    }

    void add_nlist_descendant(int i, const OctreeLevel& descendant, const std::vector<OctreeLevel>& octreeLevels) noexcept
    {
        if (descendant.is_leaf(i)) {
            _nlist.emplace_back(descendant.level(), descendant.to_leaf(i));
        } else {
            const auto& clist = descendant._clist;
            for (int inz=0; inz<clist.nnz(i); ++inz) {
                add_nlist_descendant(std::get<0>(clist.value(i, inz)), octreeLevels[descendant.level() + 1], octreeLevels);
            }
        }
    }

    void create_nlist(const std::vector<OctreeLevel>& octreeLevels) noexcept
    {
        _nlist.reserve_nrow(_n_leaf);
        for (const int i: _i_leaf) {
            ++_nlist;
            const zindex_t c = _c_node[i];
            for (int l=0; l<_level; ++l) {
                neighbor<index_t, zindex_t> neigh(l, c >> dim*(_level - l));
                do {
                    if (octreeLevels[l].has(neigh.z())) {
                        add_nlist_ancestor(octreeLevels[l].index(neigh.z()), octreeLevels[l]);
                    }
                } while (!(++neigh).end());
            }

            neighbor<index_t, zindex_t> neigh(_level, c);
            do {
                if (has(neigh.z())) {
                    add_nlist_descendant(index(neigh.z()), octreeLevels[_level], octreeLevels);
                }
            } while (!(++neigh).end());
        }
        _nlist.finish();
    }

#ifndef NDEBUG
    friend std::ostream& operator <<(std::ostream& os, const OctreeLevel& self)
    {
        return os << "level = " << self.level() <<
                     ", n_node = " << self.n_node() <<
                     ", n_leaf = " << self.n_leaf() <<
                     ", n_particle = " << self.indices().nnz() << std::endl;
    }
#endif
};


template<typename T, template<typename, size_t> typename Container>
class OctreePartitioner {
    static constexpr int dim = 3;
    using Box_t = Box<T, dim, Container>;
    using Coord_t = Container<T, dim>;
    using Node_t = OctreeNode<T, Container>;

private:
    const std::vector<Coord_t>& _positions;
    const Box_t& _box;
    std::unique_ptr<Node_t> _root;
    int _level;
    std::vector<OctreeLevel> _octreeLevels;
    std::vector<std::pair<int, int>> _partitions;
    CSRP<> _slist;

public:
    OctreePartitioner(const std::vector<Coord_t>& positions, const Box_t& box):
        _positions{positions},
        _box{box},
        _root{std::make_unique<Node_t>(0, 0, range(0, _positions.size()))}
    {

    }

    template<typename... Option_t>
    void refine(Option_t... options) noexcept
    {
        _root->refine(_positions, _box, options...);

        _level = 0;
        _root->traverse(
            [&](const Node_t* node) {
                if (node->l() > _level) {
                    _level = node->l();
                }
            }
        );

        _octreeLevels.reserve(_level + 1);
        for (int l=0; l<=_level; ++l) {
            _octreeLevels.emplace_back(l, _root.get());
            if (l >= 2) {
                _octreeLevels[l].create_ilist();
            }
        }
        for (int l=0; l<=_level - 1; ++l) {
            _octreeLevels[l].create_clist(_octreeLevels[l + 1]);
        }
        for (int l=1; l<=_level; ++l) {
            _octreeLevels[l].create_plist(_octreeLevels[l - 1]);
        }
        for (int l=0; l<=_level; ++l) {
            _octreeLevels[l].create_nlist(_octreeLevels);
        }

        _partitions.resize(_positions.size());
        for (int l=0; l<=_level; ++l) {
            const auto& olevel = _octreeLevels[l];
            const auto& indices = olevel.indices();
            for (int i_leaf=0; i_leaf<olevel.n_leaf(); ++i_leaf) {
                for (int inz=0; inz<indices.nnz(i_leaf); ++inz) {
                    _partitions[std::get<0>(indices.value(i_leaf, inz))] = {l, i_leaf};
                }
            }
        }
    }

    inline const auto& positions() const noexcept
    {
        return _positions;
    }

    inline const auto& box() const noexcept
    {
        return _box;
    }

    inline const auto& root() const noexcept
    {
        return _root;
    }

    inline int level() const noexcept
    {
        return _level;
    }

    int max_leaf_level() const noexcept
    {
        return std::max_element(_octreeLevels.cbegin(), _octreeLevels.cend(),
                                [](const auto& lhs, const auto& rhs) { return lhs.n_leaf() < rhs.n_leaf(); })->level();
    }

    inline const auto& octreeLevel(int l) const noexcept
    {
        return _octreeLevels[l];
    }

    inline const auto& get_partition(int i) const noexcept
    {
        return _partitions[i];
    }

    inline bool is_neighbor(int i, int j) const noexcept
    {
        const auto [l, i_leaf] = _partitions[i];
        return _octreeLevels[l].nlist().binary_search(i_leaf, _partitions[j]);
    }

#ifndef NDEBUG
    friend std::ostream& operator <<(std::ostream& os, const OctreePartitioner& self)
    {
        for (int l=0; l<=self.level(); ++l) {
            os << self.octreeLevel(l);
        }

        return os;
    }
#endif
};


#endif

