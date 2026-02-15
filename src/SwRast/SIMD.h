#pragma once

#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <bit>
#include <glm/mat4x4.hpp>

namespace swr {

using v_int = int32_t [[clang::ext_vector_type(16)]];
using v_uint = uint32_t [[clang::ext_vector_type(16)]];
using v_float = float [[clang::ext_vector_type(16)]];

using v_mask = uint16_t;

struct v_float4;

struct v_float2 {
    v_float x, y;

    v_float2() = default;
    v_float2(float v) { x = y = v; }
    v_float2(v_float v) { x = y = v; }
    v_float2(v_float x_, v_float y_) { x = x_, y = y_; }
    v_float2(const glm::vec2& v) { x = v.x, y = v.y; }
};
struct v_float3 {
    v_float x, y, z;

    v_float3() = default;
    v_float3(float v) { x = y = z = v; }
    v_float3(v_float v) { x = y = z = v; }
    v_float3(v_float x_, v_float y_, v_float z_) { x = x_, y = y_, z = z_; }
    v_float3(const glm::vec3& v) { x = v.x, y = v.y, z = v.z; }
    explicit v_float3(const v_float4& v);
};
struct v_float4 {
    v_float x, y, z, w;

    v_float4() = default;
    v_float4(float v) { x = y = z = w = v; }
    v_float4(v_float v) { x = y = z = w = v; }
    v_float4(v_float x_, v_float y_, v_float z_, v_float w_) { x = x_, y = y_, z = z_, w = w_; }
    v_float4(v_float3 a, v_float w_) { x = a.x, y = a.y, z = a.z, w = w_; }
    v_float4(const glm::vec4& v) { x = v.x, y = v.y, z = v.z, w = v.w; }
};

inline v_float3::v_float3(const v_float4& v) { x = v.x, y = v.y, z = v.z; }

inline v_float2 operator+(v_float2 a, v_float2 b) { return { a.x + b.x, a.y + b.y }; }
inline v_float2 operator-(v_float2 a, v_float2 b) { return { a.x - b.x, a.y - b.y }; }
inline v_float2 operator*(v_float2 a, v_float2 b) { return { a.x * b.x, a.y * b.y }; }
inline v_float2 operator/(v_float2 a, v_float2 b) { return { a.x / b.x, a.y / b.y }; }

inline v_float3 operator+(v_float3 a, v_float3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline v_float3 operator-(v_float3 a, v_float3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline v_float3 operator*(v_float3 a, v_float3 b) { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
inline v_float3 operator/(v_float3 a, v_float3 b) { return { a.x / b.x, a.y / b.y, a.z / b.z }; }

inline v_float4 operator+(v_float4 a, v_float4 b) { return { a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
inline v_float4 operator-(v_float4 a, v_float4 b) { return { a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
inline v_float4 operator*(v_float4 a, v_float4 b) { return { a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }
inline v_float4 operator/(v_float4 a, v_float4 b) { return { a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }

namespace simd {

constexpr uint32_t vec_width = 16;
constexpr v_int lane_idx = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

inline v_int load(const int32_t* ptr) { return _mm512_loadu_si512(ptr); }
inline v_int load(const uint32_t* ptr) { return _mm512_loadu_si512(ptr); }
inline v_float load(const float* ptr) { return _mm512_loadu_ps(ptr); }
inline void store(v_int x, void* ptr) { _mm512_storeu_si512(ptr, x); }
inline void store(v_float x, void* ptr) { _mm512_storeu_ps(ptr, x); }

template<int IndexScale = 4>
inline v_int gather(const int32_t* basePtr, v_int indices) {
    return _mm512_i32gather_epi32(indices, basePtr, IndexScale);
}
template<int IndexScale = 4>
inline v_float gather(const float* basePtr, v_int indices) {
    return _mm512_i32gather_ps(indices, basePtr, IndexScale);
}

inline v_mask movemask(v_int cond) {
    using vbool = bool [[clang::ext_vector_type(16)]];
    return __builtin_bit_cast(uint16_t, __builtin_convertvector(cond < 0, vbool));
}

// Pixel offsets within a 4x4 tile/fragment
//   X: [0,1,2,3, 0,1,2,3, ...]
//   Y: [0,0,0,0, 1,1,1,1, ...]
constexpr v_int FragPixelOffsetsX = simd::lane_idx & 3, FragPixelOffsetsY = simd::lane_idx >> 2;

inline v_int round2i(v_float x) { return _mm512_cvtps_epi32(x); }
inline v_int trunc2i(v_float x) { return _mm512_cvttps_epi32(x); }
inline v_float conv2f(v_int x) { return _mm512_cvtepi32_ps(x); }

inline v_int re2i(v_float x) { return _mm512_castps_si512(x); }  // reinterpret float bits to int
inline v_float re2f(v_int x) { return _mm512_castsi512_ps(x); }  // reinterpret int to float bits

inline v_int min(v_int x, v_int y) { return _mm512_min_epi32(x, y); }
inline v_int max(v_int x, v_int y) { return _mm512_max_epi32(x, y); }

inline v_float min(v_float x, v_float y) { return _mm512_min_ps(x, y); }
inline v_float max(v_float x, v_float y) { return _mm512_max_ps(x, y); }

//x * y + z
inline v_float fma(v_float x, v_float y, v_float z) { return _mm512_fmadd_ps(x, y, z); }
//x * y - z
inline v_float fms(v_float x, v_float y, v_float z) { return _mm512_fmsub_ps(x, y, z); }

// Linear interpolation between `a` and `b`: `a*(1-t) + b*t`
// https://fgiesen.wordpress.com/2012/08/15/linear-interpolation-past-present-and-future/
inline v_float lerp(v_float a, v_float b, v_float t) { return _mm512_fmadd_ps(t, b, _mm512_fnmadd_ps(t, a, a)); }

inline v_float sqrt(v_float x) { return _mm512_sqrt_ps(x); }
// approximate sqrt (14-bits precision)
inline v_float sqrt14(v_float x) { return _mm512_mul_ps(_mm512_rsqrt14_ps(x), x); }
// approximate reciprocal sqrt (14-bits precision)
inline v_float rsqrt14(v_float x) { return _mm512_rsqrt14_ps(x); }
// approximate reciprocal (14-bits precision)
inline v_float rcp14(v_float x) { return _mm512_rcp14_ps(x); }

inline v_float abs(v_float x) { return _mm512_abs_ps(x); }
inline v_int abs(v_int x) { return _mm512_abs_epi32(x); }

// lanewise: cond ? a : b
inline v_float csel(v_int cond, v_float a, v_float b) { return cond ? a : b; }
inline v_int csel(v_int cond, v_int a, v_int b) { return cond ? a : b; }

inline bool any(v_int cond) { return movemask(cond) != 0; }
inline bool all(v_int cond) { return movemask(cond) == 0xFFFF; }

// 16-bit linear interpolation with 15-bit interpolant: a + (b - a) * t
// mulhrs(a, b) = (a * b + (1 << 14)) >> 15
inline v_int lerp16(v_int a, v_int b, v_int t) { return _mm512_add_epi16(a, _mm512_mulhrs_epi16(_mm512_sub_epi16(b, a), t)); }

// shift right logical
inline v_int shrl(v_int a, uint32_t b) { return _mm512_srli_epi32(a, b); }
inline v_int shrl(v_int a, v_int b) { return _mm512_srlv_epi32(a, b); }

inline v_float dot(v_float3 a, v_float3 b) {
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z));
}
inline v_float3 normalize(v_float3 a) {
    v_float len = _mm512_rsqrt14_ps(dot(a, a));
    return { a.x * len, a.y * len, a.z * len };
}
inline v_float3 cross(v_float3 a, v_float3 b) {
    return {
        fms(a.y, b.z, a.z * b.y),
        fms(a.z, b.x, a.x * b.z),
        fms(a.x, b.y, a.y * b.x),
    };
}
inline v_float3 reflect(v_float3 i, v_float3 n) { return i - 2.0f * dot(n, i) * n; }

inline const float pi = 3.141592653589793f;
inline const float tau = 6.283185307179586f;
inline const float inv_pi = 0.3183098861837907f;

// Sleef xfastsinf_u3500()
inline v_float sin(v_float a) {
    v_int q = round2i(a * inv_pi);
    v_float d = fma(conv2f(q), -pi, a);

    v_float s = d * d;
    v_float u = -0.1881748176e-3f;
    u = fma(u, s, +0.8323502727e-2f);
    u = fma(u, s, -0.1666651368e+0f);
    u = fma(s * d, u, d);
    u = re2f(re2i(u) ^ (q << 31));  // if ((q & 1) != 0) u = -u;
    return u;
}

// Sleef xfastcosf_u3500()
inline v_float cos(v_float a) {
    v_int q = round2i(fma(a, inv_pi, -0.5f));
    v_float d = fma(conv2f(q), -pi, a - (pi * 0.5f));

    v_float s = d * d;
    v_float u = -0.1881748176e-3f;
    u = fma(u, s, +0.8323502727e-2f);
    u = fma(u, s, -0.1666651368e+0f);
    u = fma(s * d, u, d);
    u = re2f(re2i(u) ^ (~q << 31));  // if ((q & 1) != 0) u = -u;
    return u;
}

// Max relative error: sin=3.45707e-06 cos=0.00262925
inline void sincos(v_float a, v_float& rs, v_float& rc) {
    v_int q = round2i(a * inv_pi);
    v_float d = fma(conv2f(q), -pi, a);

    v_float s = d * d;
    v_float u = -0.1881748176e-3f;
    u = fma(u, s, +0.8323502727e-2f);
    u = fma(u, s, -0.1666651368e+0f);
    u = fma(s * d, u, d);

    rc = sqrt14(1.0f - u * u);
    v_int qs = (q << 31);
    rs = re2f(re2i(u) ^ qs);
    rc = re2f(re2i(rc) ^ qs);
}

// https://github.com/romeric/fastapprox/blob/master/fastapprox/src/fastlog.h
inline v_float approx_log2(v_float x) {
    v_float y = conv2f(re2i(x));
    return fma(y, 1.1920928955078125e-7f, -126.94269504f);
}
inline v_float approx_exp2(v_float x) {
    x = max(x, -126.0f);
    return re2f(round2i((1 << 23) * (x + 126.94269504f)));
}
inline v_float approx_pow(v_float x, v_float y) { return approx_exp2(approx_log2(x) * y); }

inline v_int ilog2(v_float x) {
    return (re2i(x) >> 23) - 127;  // log(x) for x <= 0 is undef, so no need to mask sign out
}

// Calculate coarse partial derivatives for a 4x4 fragment.
// https://gamedev.stackexchange.com/a/130933
inline v_float dFdx(v_float p) {
    auto a = _mm512_shuffle_ps(p, p, 0b10'10'00'00);  //[0 0 2 2]
    auto b = _mm512_shuffle_ps(p, p, 0b11'11'01'01);  //[1 1 3 3]
    return _mm512_sub_ps(b, a);
}
inline v_float dFdy(v_float p) {
    // auto a = _mm256_permute2x128_si256(p, p, 0b00'00'00'00);  // dupe lower 128 lanes
    // auto b = _mm256_permute2x128_si256(p, p, 0b01'01'01'01);  // dupe upper 128 lanes
    auto a = _mm512_shuffle_f32x4(p, p, 0b10'10'00'00);
    auto b = _mm512_shuffle_f32x4(p, p, 0b11'11'01'01);
    return _mm512_sub_ps(b, a);
}
inline v_float4 TransformVector(const glm::mat4& m, const v_float4& v) {
    return {
        m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w,
        m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w,
        m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w,
        m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w,
    };
}
inline v_float3 TransformNormal(const glm::mat3& m, const v_float3& n) {
    return {
        m[0][0] * n.x + m[1][0] * n.y + m[2][0] * n.z,
        m[0][1] * n.x + m[1][1] * n.y + m[2][1] * n.z,
        m[0][2] * n.x + m[1][2] * n.y + m[2][2] * n.z,
    };
}
inline v_float4 PerspectiveDiv(const v_float4& v) {
    v_float rw = 1.0f / v.w;
    return { v.x * rw, v.y * rw, v.z * rw, rw };
}

};  // namespace simd

template<typename T>
struct DeleteAligned {
    void operator()(T* data) const { _mm_free(data); }
};

template<typename T>
using AlignedBuffer = std::unique_ptr<T[], DeleteAligned<T>>;

template<typename T>
AlignedBuffer<T> alloc_buffer(size_t count, size_t align = 64) {
    T* ptr = (T*)_mm_malloc(count * sizeof(T), align);
    return AlignedBuffer<T>(ptr);
}

class BitIter {
    uint32_t _mask;

public:
    BitIter(uint32_t mask) : _mask(mask) {}

    BitIter& operator++() {
        _mask &= (_mask - 1);
        return *this;
    }
    uint32_t operator*() const { return (uint32_t)std::countr_zero(_mask); }
    friend bool operator!=(const BitIter& a, const BitIter& b) { return a._mask != b._mask; }

    BitIter begin() const { return *this; }
    BitIter end() const { return BitIter(0); }
};

}; // namespace swr