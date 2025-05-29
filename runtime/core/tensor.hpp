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

class Shape {
private:
    std::vector<size_t> dims_;
    size_t total_size_;

    void calculate_size() {

        total_size_ = dims_.empty() ? 0 : 
        std::accumulate(dims_.begin(), dims_.end(), 1ULL, std::multiplies<size_t>()); 

    }

public:

    // Inits

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
    }

    /*
    ** To define a sequence, you atleast need:
    ** Time Steps (which is the length of the sequence) (dims_[0])
    ** Batches per Time Step (dims_[1])
    */

} // Class Shape
} // namespace tinyquant
