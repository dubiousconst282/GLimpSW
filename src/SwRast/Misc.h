#pragma once

#include <bit>
#include <cstdint>

class BitIter {
    uint32_t _mask;

public:
    BitIter(uint32_t mask) : _mask(mask) {}

    BitIter& operator++() {
        _mask &= (_mask - 1);
        return *this;
    }
    uint32_t operator*() const { return std::countr_zero(_mask); }
    friend bool operator!=(const BitIter& a, const BitIter& b) { return a._mask != b._mask; }

    BitIter begin() const { return *this; }
    BitIter end() const { return BitIter(0); }
};