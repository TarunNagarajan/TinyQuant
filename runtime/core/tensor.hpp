#pragma once

#include <vector>
#include <memory> 
#include <cstring>
#include <stdexcept>
#include <initializer_list>
#include <cassert>
#include <algorithm>
#include <numeric>

namspace tinyquant {

class MemoryPool;

// --- START TENSOR DESCRIPTION ---

enum class DataType {
    FLOAT32, // 32 bit float
    INT8,    // 8 bit int
    INT4,    // 4 bit int
    UINT8    // 8 bit int (unsigned)
};


enum class MemoryLayout {
    ROW_MAJOR, // C-styled
    COL_MAJOR, // the format used in good old FORTRAN (mentioned in MIT 6.172)
    BLOCKED    // Blocked layout -> cache optimization
};

// --- END OF TENSOR DESCRIPTION ---

// --- START SHAPE CLASS --

class Shape {
private:
    std::vector<size_t> dims_;
    size_t total_size_;

    void calculate_size() {

        total_size_ = dims_.empty() ? 0 : 
        std::accumulate(dims_.begin(), dims_.end(), 1ULL, std::multiplies<size_t>()); 

    }

public:

    // CONSTRUCTORS

    Shape() : total_size_(0) {}

    Shape(std:::initializer_list<size_t> dims) : dims_(dims) {
        calculate_size();
    }

    Shape(const std::vector<size_t> dims) : dims_(dims) {
        calculate_size();
    }

    // Dimension Access

    size_t operator[](size_t index) const {
        if (index >= dims_.size()) {
            throw std::out_of_range("OUT OF BOUNDS: SHAPE INDEX");
        }
        return dims_[index];
    }


    size_t& operator[](size_t index) {
        if (index >= dims_.size()) {
            throw std::out_of_range("OUT OF BOUNDS: SHAPE INDEX");
        }
        return dims_[index];
    }

    size_t ndim() const {
        return dims_.size();
    }

    size_t size() const {
        return total_size_;
    }

    const std::vector<size_t>& dims() const {
        return dims_;
    }

    // Reshaping 

    Shape reshape(const std::vector<size_t>& new_dim) const {
        size_t new_size = std::accumulate(new_dim.begin(), new_dim.end(), 1ULL, std::multiplies<size_t>());
        if (new_size != total_size_) {
            throw std::invalid_argument("SIZE MISMATCH: CANNOT RESHAPE")
        }
        return Shape(new_dim); 
    }

    // @Feature: RNN Support

    bool is_sequence() const {
        return dims_.size() >= 2; 
        // returns true, if the tensor has a sequence dimension
    }

    /*
    ** To define a sequence, you atleast need:
    ** Time Steps (which is the length of the sequence) (dims_[0])
    ** Batches per Time Step (dims_[1])

    ** Sequence of Batches is the standard format for RNN/Sequence Model inputs. 
    */

    size_t sequence_length() const {
        return is_sequence() ? dims_[0] : 1; 
        // returns the time steps (length of sequence), if it is a sequence.
    }

    size_t batch_size() const {
        return is_sequence() && dims_.size() >= 2 ? dims_[1] : 1;
        // returns the number of batches per time step (dims_[1]), if it is a sequence.
    }

    size_t feature_size() const {
        if (dims_.size() == 1) return dims_[0];
        if (dims_.size() == 2) return dims_[1];
        if (dims_.size() == 3) return dims_[2];

        return std::accumulate(dims_begin() + 2, dims_.end(), 1ULL, std::multiplies<size_t>());
    }

    bool operator==(const Shape& other) const {
        return dims_ == other.dims_;
    }

    bool operator!=(const Shape& other) const {
        return !(*this == other);
        // return true if this and other are NOT equal, and false if they ARE equal.
    }
}; // --- END SHAPE CLASS ---

// --- START TENSOR CLASS ---
class Tensor {
private:
    Shape shape_;
    DataType dtype_;
    MemoryLayout layout_;
    std::shared_ptr<void> data_; 
    size_t byte_size_;
    bool owns_data_; 

    size_t bytes_per_element() const {
        switch(dtype_) {
            case DataType::FLOAT32: return 4;
            case DataType::INT8: return 1;
            case DataType::INT4: return 1;
            case DataType::UINT8: return 1; // @IDK Packed, but 1 byte min
        }
    }

    static constexpr size_t ALIGNMENT = 32;
    // compiler-time constant shared along all instances of the Tensor class.

    void* aligned_alloc(size_t size) {
        void* ptr = nullptr;
        #ifdef _WIN32
            ptr = _aligned_malloc(size, ALIGNMENT);
        #else
            if (posix_memalign(&ptr, ALIGNMENT, size) != 0) ptr = nullptr;
        #endif

        if (!ptr) {
            throw std::bad_alloc();
        }

        return ptr;
    }

    void aligned_free(void *ptr) {
        if (ptr) {
            #ifdef _WIN32 
                _aligned_free(ptr);
            #else
                free(ptr);
            #endif
        }   
    }

public:

    // CONSTRUCTORS 
    Tensor() : dtype_(DataType::FLOAT32), layout_(MemoryLayout::ROW_MAJOR), byte_size_(0), owns_data_(true) {}

    Tensor(const Shape& shape, DataType dtype = DataType::FLOAT32,
           MemoryLayout layout = MemoryLayout::ROW_MAJOR)
           : shape_(shape), dtype_(dtype), layout_(layout), owns_data_(true) {
            byte_size_ = shape_.size() * bytes_per_element(); 

            if (byte_size_ > 0) {
                void* raw_ptr = aligned_alloc(byte_size_);
                data_ = std::shared_ptr<void>(raw_ptr, [this](void* p) {
                    aligned_free(p);
                });

                std::memset(raw_ptr, 0, byte_size_);
            }
        }

    // + INITIALIZER LIST FOR TESTING

    template<typename T>
    Tensor(const Shape& shape, std::initializer_list<T> values, DataType dtype = DataType::FLOAT32) 
    : Tensor(shape, dtype) {
        if (values.size() != shapes_.size()) {
            throw std::invalid_argument("SIZE MISMATCH: INITIALIZER LIST & TENSOR SHAPE");  
        }

        T* ptr = static_cast<T*>(data_.get());
        std::copy(values.begin(), values.end(), ptr);
    }

    Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), layout_(other.layout_),
    byte_size_(other.byte_size_), owns_data_(true) {
        if (byte_size_ > 0) {
            void* raw_ptr = aligned_alloc(byte_size_);
            data_ = std::shared_ptr<void>(raw_ptr, [this](void* p) {
                aligned_free(p);
            });

            std::memcpy(raw_ptr,other.data_.get(),byte_size_);
        }
    }

    Tensor(Tensor&& other) noexcept

}
// --- END TENSOR CLASS ---
} // namespace tinyquant
