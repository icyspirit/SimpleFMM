
#ifndef ALIGNED_HPP
#define ALIGNED_HPP


#include <cstdlib>
#include <new>
#include <vector>

 
template<typename T, std::size_t alignment>
struct AlignedAllocator {
    using value_type = T;
 
    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, alignment>;
    };
 
    [[nodiscard]] static T* allocate(std::size_t n)
    {
        void* p = std::aligned_alloc(alignment, ((sizeof(T)*n + alignment - 1)/alignment)*alignment);
        if (!p) {
            throw std::bad_alloc();
        }

        return reinterpret_cast<T*>(p);
    }
 
    static void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept
    {
        std::free(p);
    }
};
 

template<typename T, std::size_t alignment=64>
using avector = std::vector<T, AlignedAllocator<T, alignment>>;
 

#endif

