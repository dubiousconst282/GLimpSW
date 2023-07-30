#pragma once

#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <glm/mat4x4.hpp>

namespace swr {

struct VInt {
    static const uint32_t Length = sizeof(__m512i) / sizeof(int32_t);

    __m512i reg;

    VInt() { reg = _mm512_setzero_epi32(); }
    VInt(__m512i x) { reg = x; }
    VInt(int32_t x) { reg = _mm512_set1_epi32(x); }
    inline operator __m512i() { return reg; }

    inline int32_t& operator[](size_t idx) const {
        assert(idx >= 0 && idx < Length);
        return ((int32_t*)&reg)[idx];
    }

    static inline VInt load(const int32_t* ptr) { return _mm512_loadu_si512((__m512i*)ptr); }
    inline void store(int32_t* ptr) const { _mm512_storeu_si512((__m512i*)ptr, reg); }

    template<int IndexScale = 1>
    static inline VInt gather(const int32_t* basePtr, VInt indices) { return _mm512_i32gather_epi32(indices, basePtr, IndexScale); }

    static inline VInt ramp() { return _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); }
};
struct VFloat {
    static const uint32_t Length = sizeof(__m512) / sizeof(float);

    __m512 reg;

    VFloat() { reg = _mm512_setzero_ps(); }
    VFloat(__m512 x) { reg = x; }
    VFloat(float x) { reg = _mm512_set1_ps(x); }
    inline operator __m512() { return reg; }

    inline float& operator[](size_t idx) const {
        assert(idx >= 0 && idx < Length);
        return ((float*)&reg)[idx];
    }

    static inline VFloat load(const float* ptr) { return _mm512_loadu_ps(ptr); }
    inline void store(float* ptr) const { _mm512_storeu_ps(ptr, reg); }

    template<int IndexScale = 1>
    static inline VFloat gather(const float* basePtr, VInt indices) { return _mm512_i32gather_ps(indices.reg, basePtr, IndexScale); }

    static inline VFloat ramp() { return _mm512_setr_ps(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); }
};

struct VFloat3 {
    VFloat x, y, z;
};
struct VFloat4 {
    VFloat x, y, z, w;
};

#define _SIMD_DEF_OPERATORS(TVec, OpSuffix, MulOp, BitSuffix)                      \
    inline TVec operator+(TVec a, TVec b) { return _mm512_add_##OpSuffix(a, b); }  \
    inline TVec operator-(TVec a, TVec b) { return _mm512_sub_##OpSuffix(a, b); }  \
    inline TVec operator*(TVec a, TVec b) { return _mm512_##MulOp(a, b); }         \
    inline TVec operator&(TVec a, TVec b) { return _mm512_and_##BitSuffix(a, b); } \
    inline TVec operator|(TVec a, TVec b) { return _mm512_or_##BitSuffix(a, b); }  \
    inline TVec operator^(TVec a, TVec b) { return _mm512_xor_##BitSuffix(a, b); } \
                                                                                   \
    inline TVec operator+=(TVec& a, TVec b) { return a = (a + b); }                \
    inline TVec operator-=(TVec& a, TVec b) { return a = (a - b); }                \
    inline TVec operator*=(TVec& a, TVec b) { return a = (a * b); }

_SIMD_DEF_OPERATORS(VFloat, ps, mul_ps, ps);
inline VFloat operator/(VFloat a, VFloat b) { return _mm512_div_ps(a, b); }

_SIMD_DEF_OPERATORS(VInt, epi32, mullo_epi32, si512);
inline VInt operator>>(VInt a, uint32_t b) { return _mm512_srai_epi32(a, b); }
inline VInt operator<<(VInt a, uint32_t b) { return _mm512_slli_epi32(a, b); }

inline VFloat operator-(VFloat a) { return a ^ -0.0f; }

namespace simd {

inline VInt round2i(VFloat x) { return _mm512_cvtps_epi32(x.reg); }
inline VInt trunc2i(VFloat x) { return _mm512_cvttps_epi32(x.reg); }
inline VFloat conv2f(VInt x) { return _mm512_cvtepi32_ps(x.reg); }

inline VInt re2i(VFloat x) { return _mm512_castps_si512(x); }  // reinterpret float bits to int
inline VFloat re2f(VInt x) { return _mm512_castsi512_ps(x); }  // reinterpret int to float bits

inline VInt min(VInt x, VInt y) { return _mm512_min_epi32(x, y); }
inline VInt max(VInt x, VInt y) { return _mm512_max_epi32(x, y); }

inline VFloat min(VFloat x, VFloat y) { return _mm512_min_ps(x, y); }
inline VFloat max(VFloat x, VFloat y) { return _mm512_max_ps(x, y); }

//x * y + z
inline VFloat fma(VFloat x, VFloat y, VFloat z) { return _mm512_fmadd_ps(x, y, z); }
//x * y - z
inline VFloat fms(VFloat x, VFloat y, VFloat z) { return _mm512_fmsub_ps(x, y, z); }

// 16-bit linear interpolation with 15-bit interpolant: a + (b - a) * t
// mulhrs(a, b) = (a * b + (1 << 14)) >> 15
inline VInt lerp16(VInt a, VInt b, VInt t) { return _mm512_add_epi16(a, _mm512_mulhrs_epi16(_mm512_sub_epi16(b, a), t)); }

// shift right logical
inline VInt shrl(VInt a, uint32_t b) { return _mm512_srli_epi32(a, b); }
inline VInt shrl(VInt a, VInt b) { return _mm512_srlv_epi32(a, b); }

inline VFloat dot(VFloat3 a, VFloat3 b) {
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z));
}
inline VFloat3 normalize(VFloat3 a) {
    VFloat len = _mm512_rsqrt14_ps(dot(a, a));
    return { a.x * len, a.y * len, a.z * len };
}
inline VFloat3 cross(VFloat3 a, VFloat3 b) {
    return {
        fms(a.y, b.z, a.z * b.y),
        fms(a.z, b.x, a.x * b.z),
        fms(a.x, b.y, a.y * b.x),
    };
}

inline const float pi = 3.141592653589793f;
inline const float inv_pi = 0.3183098861837907f;

// Sleef xfastsinf_u3500()
inline VFloat sin(VFloat a) {
    VInt q = round2i(a * inv_pi);
    VFloat d = fma(conv2f(q), -pi, a);

    VFloat s = d * d;
    VFloat u = -0.1881748176e-3f;
    u = fma(u, s, +0.8323502727e-2f);
    u = fma(u, s, -0.1666651368e+0f);
    u = fma(s * d, u, d);
    u = u ^ re2f(shrl(q, 31));  // if ((q & 1) != 0) u = -u;
    return u;
}

// Sleef xfastcosf_u3500()
static VFloat cos(VFloat a) {
    VInt q = round2i(fma(a, inv_pi, -0.5f));
    VFloat d = fma(conv2f(q), -pi, a - (pi * 0.5f));

    VFloat s = d * d;
    VFloat u = -0.1881748176e-3f;
    u = fma(u, s, +0.8323502727e-2f);
    u = fma(u, s, -0.1666651368e+0f);
    u = fma(s * d, u, d);
    u = u ^ re2f(shrl(~0 ^ q, 31));  // if ((q & 1) == 0) u = -u;
    return u;
}

// https://github.com/romeric/fastapprox/blob/master/fastapprox/src/fastlog.h
static VFloat approx_log2(VFloat x) {
    VFloat y = conv2f(re2i(x));
    return fma(y, 1.1920928955078125e-7f, -126.94269504f);
}

inline VInt PackRGBA(const VFloat4& color) {
    auto ri = _mm512_cvtps_epi32(color.x * 255.0f);
    auto gi = _mm512_cvtps_epi32(color.y * 255.0f);
    auto bi = _mm512_cvtps_epi32(color.z * 255.0f);
    auto ai = _mm512_cvtps_epi32(color.w * 255.0f);

    auto rg = _mm512_packs_epi32(ri, gi);
    auto ba = _mm512_packs_epi32(bi, ai);
    auto cb = _mm512_packus_epi16(rg, ba);

    auto shuffMask = _mm512_setr4_epi32(0x0C'08'04'00, 0x0D'09'05'01, 0x0E'0A'06'02, 0x0F'0B'07'03);  // 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
    return _mm512_shuffle_epi8(cb, shuffMask);
}

inline VFloat4 TransformVector(const glm::mat4& m, const VFloat4& v) {
    return {
        m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
        m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w,
        m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w,
        m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w,
    };
}

};  // namespace simd

}; // namespace swr