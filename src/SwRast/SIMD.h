#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>

#include <glm/glm.hpp>

#include <immintrin.h>

#define SIMD_INLINE [[gnu::always_inline, gnu::nodebug]] inline

namespace simd {

// Default vector width
constexpr int vec_width = sizeof(__m512i) / sizeof(int32_t);

template<typename E, int N = vec_width>
using vec = E [[clang::ext_vector_type(N)]];

template<typename T>
using elem_type = decltype(+T{}[0]);

template<typename T>
concept is_vector = requires { __builtin_vectorelements(T); };

template<is_vector T>
constexpr int width_of = __builtin_vectorelements(T);

template<is_vector T>
using mask_vec = decltype(T{} == T{});

template<is_vector T>
using bitmask = std::conditional_t<width_of<T> <= 8, uint8_t,   //
                std::conditional_t<width_of<T> <= 16, uint16_t, //
                std::conditional_t<width_of<T> <= 32, uint32_t, uint64_t>>>;

// Operators
#define SIMD_DEF_OPS    \
    SIMD_DEF_BIN_OP(+)  \
    SIMD_DEF_BIN_OP(-)  \
    SIMD_DEF_BIN_OP(*)  \
    SIMD_DEF_BIN_OP(/)  \
                        \
    SIMD_DEF_BIN_OP(|)  \
    SIMD_DEF_BIN_OP(&)  \
    SIMD_DEF_BIN_OP(^)  \
    SIMD_DEF_BIN_OP(<<) \
    SIMD_DEF_BIN_OP(>>)

template<typename T, int N>
struct mdvec;

template<typename T>
struct mdvec<T, 2> {
    T x, y;

    SIMD_INLINE constexpr mdvec() = default;
    SIMD_INLINE constexpr mdvec(T v_) : x(v_), y(v_) {}
    SIMD_INLINE constexpr mdvec(T x_, T y_) : x(x_), y(y_) {}
    SIMD_INLINE constexpr mdvec(const glm::vec<2, elem_type<T>>& v_) : x(v_.x), y(v_.y) {}
    SIMD_INLINE explicit constexpr mdvec(const mdvec<T, 3>& v_) : x(v_.x), y(v_.y) {}
    SIMD_INLINE explicit constexpr mdvec(const mdvec<T, 4>& v_) : x(v_.x), y(v_.y) {}

    SIMD_INLINE T& operator[](size_t idx) {
        assert(idx < 2);
        return idx == 0 ? x : y;
    }

#undef SIMD_DEF_BIN_OP
#define SIMD_DEF_BIN_OP(sym) \
    SIMD_INLINE constexpr mdvec friend operator sym(mdvec a, mdvec b) { return { a.x sym b.x, a.y sym b.y }; } \
    SIMD_INLINE constexpr friend mdvec& operator sym##=(mdvec& a, mdvec b) { return a = (a sym b); }

    SIMD_DEF_OPS;
};
template<typename T>
struct mdvec<T, 3> {
    T x, y, z;

    SIMD_INLINE constexpr mdvec() = default;
    SIMD_INLINE constexpr mdvec(T v_) : x(v_), y(v_), z(v_) {}
    SIMD_INLINE constexpr mdvec(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    SIMD_INLINE constexpr mdvec(mdvec<T, 2> v_, T z_) : x(v_.x), y(v_.y), z(z_) {}
    SIMD_INLINE constexpr mdvec(const glm::vec<3, elem_type<T>>& v_) : x(v_.x), y(v_.y), z(v_.z) {}
    SIMD_INLINE explicit constexpr mdvec(const mdvec<T, 4>& v_) : x(v_.x), y(v_.y), z(v_.z) {}

    SIMD_INLINE T& operator[](size_t idx) {
        assert(idx < 3);
        return idx == 0 ? x : idx == 1 ? y : z;
    }
#undef SIMD_DEF_BIN_OP
#define SIMD_DEF_BIN_OP(sym) \
    SIMD_INLINE constexpr friend mdvec operator sym(mdvec a, mdvec b) { return { a.x sym b.x, a.y sym b.y, a.z sym b.z }; } \
    SIMD_INLINE constexpr friend mdvec& operator sym##=(mdvec& a, mdvec b) { return a = (a sym b); }

    SIMD_DEF_OPS;
};

template<typename T>
struct mdvec<T, 4> {
    T x, y, z, w;

    SIMD_INLINE constexpr mdvec() = default;
    SIMD_INLINE constexpr mdvec(T v_) : x(v_), y(v_), z(v_), w(v_) {}
    SIMD_INLINE constexpr mdvec(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}
    SIMD_INLINE constexpr mdvec(mdvec<T, 2> v_, T z_, T w_) : x(v_.x), y(v_.y), z(z_), w(w_) {}
    SIMD_INLINE constexpr mdvec(mdvec<T, 3> v_, T w_) : x(v_.x), y(v_.y), z(v_.z), w(w_) {}
    SIMD_INLINE constexpr mdvec(const glm::vec<4, elem_type<T>>& v_) : x(v_.x), y(v_.y), z(v_.z), w(v_.w) {}

    SIMD_INLINE T& operator[](size_t idx) {
        assert(idx < 4);
        return idx == 0 ? x : idx == 1 ? y : idx == 2 ? z : w;
    }
#undef SIMD_DEF_BIN_OP
#define SIMD_DEF_BIN_OP(sym) \
    SIMD_INLINE constexpr mdvec friend operator sym(mdvec a, mdvec b) { return { a.x sym b.x, a.y sym b.y, a.z sym b.z, a.w sym b.w }; } \
    SIMD_INLINE constexpr friend mdvec& operator sym##=(mdvec& a, mdvec b) { return a = (a sym b); }

    SIMD_DEF_OPS;
};
#undef SIMD_DEF_BIN_OP
#undef SIMD_DEF_OPS

};  // namespace simd

// Type aliases
using v_int = simd::vec<int32_t>;
using v_uint = simd::vec<uint32_t>;
using v_float = simd::vec<float>;

using v_mask = simd::bitmask<v_int>;

using v_float2 = simd::mdvec<v_float, 2>;
using v_float3 = simd::mdvec<v_float, 3>;
using v_float4 = simd::mdvec<v_float, 4>;

using v_int2 = simd::mdvec<v_int, 2>;
using v_int3 = simd::mdvec<v_int, 3>;
using v_int4 = simd::mdvec<v_int, 4>;

using v_uint2 = simd::mdvec<v_uint, 2>;
using v_uint3 = simd::mdvec<v_uint, 3>;
using v_uint4 = simd::mdvec<v_uint, 4>;

// TODO: maybe nuke GLM with ext_vector_type
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;
using float4x4 = glm::mat4x4;

using int2 = glm::ivec2;
using int3 = glm::ivec3;
using int4 = glm::ivec4;

using uint2 = glm::uvec2;
using uint3 = glm::uvec3;
using uint4 = glm::uvec4;

namespace simd {

#define SIMD_TFN_ANY template<is_vector T> SIMD_INLINE
#define SIMD_TFN_INT   template<typename T> requires(is_vector<T> && std::is_integral_v<elem_type<T>>) SIMD_INLINE
#define SIMD_TFN_FLOAT template<typename T> requires(is_vector<T> && std::is_floating_point_v<elem_type<T>>) SIMD_INLINE
#define SIMD_TFN_MASK  template<typename T> requires(is_vector<T> && std::is_signed_v<elem_type<T>>) SIMD_INLINE

// Sequence of lane indices: [0, 1, 2, 3, ...]
template<is_vector T>
constexpr T lane_idx = (                              //
    []<size_t... Idx>(std::index_sequence<Idx...>) {  //
        return T{ (elem_type<T>)Idx... };             //
    })(std::make_index_sequence<width_of<T>>());

//////////////////////////////////////// Memory ////////////////////////////////////////

template<typename T>
SIMD_INLINE T load(const void* ptr) {
    // This minimizes spills at -O0 compared to memcpy.
    struct [[gnu::packed, gnu::may_alias]] load_buf { T v; };
    return ((const load_buf*)ptr)->v;
}
template<typename E, int N = sizeof(__m512) / sizeof(E)>
SIMD_INLINE vec<E, N> load(const E* ptr) {
    return load<vec<E, N>>(ptr);
}

template<typename T>
SIMD_INLINE void store(void* ptr, const T& value) {
    struct [[gnu::packed, gnu::may_alias]] load_buf { T v; };
    ((load_buf*)ptr)->v = value;
}

SIMD_TFN_ANY T gather(const void* ptr, mask_vec<T> idx, vec<bool, width_of<T>> mask = true) {
    return __builtin_masked_gather(mask, idx, (const elem_type<T>*)ptr);
}
template<typename E, int N = sizeof(__m512) / sizeof(E)>
SIMD_INLINE vec<E, N> gather(const E* ptr, mask_vec<vec<E, N>> idx, vec<bool, N> mask = true) {
    return __builtin_masked_gather(mask, idx, ptr);
}

// Gather variants optimized for small arrays
SIMD_INLINE v_uint gather_preload64(const uint32_t data[64], v_uint idx) {
    v_uint v0 = _mm512_load_si512(&data[0]);
    v_uint v1 = _mm512_load_si512(&data[16]);
    v_uint v2 = _mm512_load_si512(&data[32]);
    v_uint v3 = _mm512_load_si512(&data[48]);
    v_uint p01 = _mm512_permutex2var_epi32(v0, idx, v1);
    v_uint p23 = _mm512_permutex2var_epi32(v2, idx, v3);
    return idx < 32 ? p01 : p23;
}
SIMD_INLINE v_float gather_preload64(const float data[64], v_uint idx) {
    return __builtin_bit_cast(v_float, gather_preload64((const uint32_t*)data, idx));
}
// Gather 64 bytes from 128-byte array
SIMD_INLINE v_uint gather_preload128(const uint8_t data[128], v_uint idx) {
    v_uint v0 = _mm512_load_si512(&data[0]);
    v_uint v1 = _mm512_load_si512(&data[64]);
    return _mm512_permutex2var_epi8(v0, idx, v1);
}

// Shuffle by constant indices
template<auto indices, is_vector T> requires(std::is_integral_v<elem_type<decltype(indices)>>)
SIMD_INLINE constexpr T shuffle(T values) {
    const auto do_shuffle = [&]<size_t... Idx>(std::index_sequence<Idx...>) {
        return __builtin_shufflevector(values, values, indices[Idx]...);
    };
    return do_shuffle(std::make_index_sequence<width_of<T>>());
}

// Narrow lane mask to scalar bitmask, where each bit is set if `mask[i] < 0` (SSE behavior).
SIMD_TFN_MASK constexpr bitmask<T> movemask(T mask) {
    return __builtin_bit_cast(bitmask<T>, __builtin_convertvector(mask < 0, vec<bool, width_of<T>>));
}

// Checks if any lane of the given mask vector is true.
SIMD_TFN_MASK constexpr bool any(T mask) { return movemask(mask) != 0; }

// Checks if all lanes of the given mask vector are true.
SIMD_TFN_MASK constexpr bool all(T mask) { return movemask(mask) == (unsigned _BitInt(width_of<T>))(-1); }

template<is_vector T>
SIMD_INLINE void cmov(T& dest, T ifTrue, mask_vec<T> cond) {
    dest = cond ? ifTrue : dest;
}
template<typename T, int N>
SIMD_INLINE void cmov(mdvec<T, N>& dest, mdvec<T, N> ifTrue, mask_vec<T> cond) {
    for (uint32_t i = 0; i < N; i++) {
        dest[i] = cond ? ifTrue[i] : dest[i];
    }
}

template<typename T, int N>
SIMD_INLINE constexpr mdvec<T, N> select(mask_vec<T> cond, mdvec<T, N> ifTrue, mdvec<T, N> ifFalse) {
    mdvec<T, N> r;
    for (uint32_t i = 0; i < N; i++) {
        r[i] = cond ? ifTrue[i] : ifFalse[i];
    }
    return r;
}
template<typename T, int N>
SIMD_INLINE constexpr mdvec<T, N> select(mdvec<mask_vec<T>, N> cond, mdvec<T, N> ifTrue, mdvec<T, N> ifFalse) {
    mdvec<T, N> r;
    for (uint32_t i = 0; i < N; i++) {
        r[i] = cond[i] ? ifTrue[i] : ifFalse[i];
    }
    return r;
}
//////////////////////////////////////// Fundamentals ////////////////////////////////////////

// a * b + c
SIMD_TFN_FLOAT T fma(T a, T b, T c) { return __builtin_elementwise_fma(a, b, c); }
SIMD_TFN_FLOAT T fma(T a, elem_type<T> b, T c) { return __builtin_elementwise_fma(a, T(b), c); }
SIMD_TFN_FLOAT T fma(T a, T b, elem_type<T> c) { return __builtin_elementwise_fma(a, b, T(c)); }
SIMD_TFN_FLOAT T fma(T a, elem_type<T> b, elem_type<T> c) { return __builtin_elementwise_fma(a, T(b), T(c)); }

SIMD_TFN_FLOAT T sqrt(T x) { return __builtin_elementwise_sqrt(x); }
SIMD_INLINE v_float approx_rsqrt(v_float x) { return _mm512_rsqrt14_ps(x); }
SIMD_INLINE v_float approx_rcp(v_float x) { return _mm512_rcp14_ps(x); }
SIMD_INLINE v_float approx_sqrt(v_float x) { return approx_rsqrt(x) * x; }

SIMD_TFN_FLOAT T floor(T x) { return __builtin_elementwise_floor(x); }
SIMD_TFN_FLOAT T ceil(T x) { return __builtin_elementwise_ceil(x); }
SIMD_TFN_FLOAT T round(T x) { return __builtin_elementwise_round(x); }
SIMD_INLINE v_float fract(v_float x) { return _mm512_reduce_ps(x, _MM_FROUND_TO_NEG_INF); }

SIMD_INLINE v_int floor2i(v_float x) { return _mm512_cvt_roundps_epi32(x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); }
SIMD_INLINE v_int round2i(v_float x) { return _mm512_cvtps_epi32(x); }

SIMD_TFN_ANY T abs(T x) { return __builtin_elementwise_abs(x); }

SIMD_TFN_ANY T min(T x, T y) {
    if constexpr (std::is_integral_v<elem_type<T>>) {
        return __builtin_elementwise_min(x, y);
    } else {
        // use -ffast-math to get `vpminps` on x86.
        return __builtin_elementwise_minnum(x, y);
    }
}
SIMD_TFN_ANY T min(T x, elem_type<T> y) { return min(x, T(y)); }
SIMD_TFN_ANY T min(elem_type<T> x, T y) { return min(T(x), y); }

SIMD_TFN_ANY T max(T x, T y) {
    if constexpr (std::is_integral_v<elem_type<T>>) {
        return __builtin_elementwise_max(x, y);
    } else {
        // use -ffast-math to get `vpmaxps` on x86.
        return __builtin_elementwise_maxnum(x, y);
    }
}
SIMD_TFN_ANY T max(T x, elem_type<T> y) { return max(x, T(y)); }
SIMD_TFN_ANY T max(elem_type<T> x, T y) { return max(T(x), y); }

SIMD_TFN_ANY T clamp(T x, T a, T b) { return min(max(x, a), b); }
SIMD_TFN_ANY T clamp(T x, elem_type<T> a, elem_type<T> b) { return min(max(x, T(a)), T(b)); }

// min(abs(x), abs(y))
SIMD_INLINE v_float min_abs(v_float x, v_float y) { return _mm512_range_ps(x, y, 0b10'10); }
SIMD_INLINE v_float max_abs(v_float x, v_float y) { return _mm512_range_ps(x, y, 0b10'11); }

SIMD_TFN_INT T add_sat(T x, T y) { return __builtin_elementwise_add_sat(x, y); }
SIMD_TFN_INT T sub_sat(T x, T y) { return __builtin_elementwise_sub_sat(x, y); }

// y < 0 ? -x : x
SIMD_INLINE v_float mulsign(v_float x, v_float y) {
    // return as<float>(as<uint32_t>(x) ^ (as<uint32_t>(y) & 0x8000'0000));
    return _mm512_ternarylogic_epi32(x, y, v_uint(0x7FFF'FFFF), _MM_TERNLOG_A ^ (_MM_TERNLOG_B & ~_MM_TERNLOG_C));
}
// s < 0 ? -abs(x) : abs(x)
SIMD_INLINE v_float copysign(v_float x, v_float s) {
    // return as<float>((as<uint32_t>(x) & 0x7FFF'FFFF) | (as<uint32_t>(s) & 0x8000'0000));
    return _mm512_ternarylogic_epi32(x, s, v_uint(0x7FFF'FFFF), (_MM_TERNLOG_A & _MM_TERNLOG_C) | (_MM_TERNLOG_B & ~_MM_TERNLOG_C));
}

template<typename R, is_vector T>
SIMD_INLINE constexpr vec<R, width_of<T>> conv(T x) {
    return __builtin_convertvector(x, vec<R, width_of<T>>);
}

template<typename R, is_vector T, int N>
SIMD_INLINE constexpr mdvec<vec<R, width_of<T>>, N> conv(mdvec<T, N> x) {
    mdvec<vec<R, width_of<T>>, N> r;
    for (uint32_t i = 0; i < N; i++) {
        r[i] = __builtin_convertvector(x[i], vec<R, width_of<T>>);
    }
    return r;
}

template<std::integral R, is_vector T> requires(sizeof(R) < sizeof(elem_type<T>))
SIMD_INLINE constexpr vec<R, width_of<T>> conv_sat(T x) {
    using E = elem_type<T>;

    x = __builtin_elementwise_max(x, T(static_cast<E>(std::is_unsigned_v<E> ? 0 : std::numeric_limits<R>::min())));
    x = __builtin_elementwise_min(x, T(static_cast<E>(std::numeric_limits<R>::max())));
    return __builtin_convertvector(x, vec<R, width_of<T>>);
}

template<typename Dst, typename Src>
SIMD_INLINE constexpr Dst as(Src x) { return __builtin_bit_cast(Dst, x); }

template<typename Dst, typename Src, int N>
SIMD_INLINE vec<Dst, sizeof(Src) * N / sizeof(Dst)> as(vec<Src, N> x) {
    return __builtin_bit_cast(vec<Dst, sizeof(Src) * N / sizeof(Dst)>, x);
}

//////////////////////////////////////// Math ////////////////////////////////////////

constexpr float pi = 3.141592653589793f;
constexpr float tau = 6.283185307179586f;
constexpr float inv_pi = 0.3183098861837907f;

// Approximates sin(2πx) and cos(2πx)
// Max abs error: sin=9.08735e-07 cos=7.07177e-06
// https://publik-void.github.io/sin-cos-approximations
SIMD_INLINE v_float2 sincos_2pi(v_float x) {
    // Reduce range to -1/4..1/4
    //   x = x + 0.25
    //   x = abs(x - floor(x + 0.5)) - 0.25
#if __AVX512DQ__
    x = _mm512_reduce_ps(x + 0.25f, _MM_FROUND_TO_NEAREST_INT);
#else
    x = x + 0.25f;
    x = x - round(x);
#endif
    v_float x1 = abs(x) - 0.25f;
    v_float x2 = x1 * x1;

    v_float r_sin = fma(x2, fma(x2, fma(x2, -70.993433272f, 81.340768887f), -41.337142371f), 6.283164044f) * x1;
    v_float r_cos = fma(x2, fma(x2, fma(x2, -78.216131988f, 64.660541218f), -19.735752060f), 0.999993295f);
    r_cos = mulsign(r_cos, x);

    return v_float2(r_sin, r_cos);
}

SIMD_INLINE v_float2 sincos(v_float a) { return sincos_2pi(a * 0.15915494309189535f); }
SIMD_INLINE v_float sin(v_float a) { return sincos(a).x; }
SIMD_INLINE v_float cos(v_float a) { return sincos(a).y; }

// https://github.com/romeric/fastapprox/blob/master/fastapprox/src/fastlog.h
SIMD_INLINE v_float approx_log2(v_float x) {
    v_float y = conv<float>(as<uint32_t>(x));
    return fma(y, 1.1920928955078125e-7f, -126.94269504f);
}
SIMD_INLINE v_float approx_exp2(v_float x) {
    x = max(x, -126.0f);
    return as<float>(round2i((1 << 23) * (x + 126.94269504f)));
}
SIMD_INLINE v_float approx_pow(v_float x, v_float y) { return approx_exp2(approx_log2(x) * y); }

SIMD_INLINE v_int ilog2(v_float x) { return (as<v_int>(x) - (127 << 23)) >> 23; }

SIMD_INLINE v_float dot(v_float2 a, v_float2 b) { return fma(a.x, b.x, a.y * b.y); }
SIMD_INLINE v_float dot(v_float3 a, v_float3 b) { return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z)); }

SIMD_INLINE v_float3 cross(v_float3 a, v_float3 b) {
    return {
        fma(a.y, b.z, -a.z * b.y),
        fma(a.z, b.x, -a.x * b.z),
        fma(a.x, b.y, -a.y * b.x),
    };
}

SIMD_INLINE v_float3 normalize(v_float3 a) { return a * approx_rsqrt(dot(a, a)); }
SIMD_INLINE v_float length(v_float3 p) { return approx_sqrt(dot(p, p)); }

SIMD_INLINE v_float3 reflect(v_float3 i, v_float3 n) { return i - 2.0f * dot(n, i) * n; }

// Linear interpolation between `a` and `b`: `a*(1-t) + b*t`
// https://fgiesen.wordpress.com/2012/08/15/linear-interpolation-past-present-and-future/
SIMD_TFN_FLOAT T lerp(T a, T b, T t) { return fma(t, b, fma(-t, a, a)); }

// 16-bit linear interpolation with 15-bit interpolant: a + (b - a) * t
// mulhrs(a, b) = (a * b + (1 << 14)) >> 15
SIMD_INLINE v_uint lerp16(v_uint a, v_uint b, v_uint t) { return _mm512_add_epi16(a, _mm512_mulhrs_epi16(_mm512_sub_epi16(b, a), t)); }

SIMD_INLINE v_float smoothstep(v_float a, v_float b, v_float t) {
    t = clamp((t - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

SIMD_INLINE v_float4 mul(const glm::mat4& m, const v_float4& v) {
    return {
        fma(v.x, m[0][0], fma(v.y, m[1][0], fma(v.z, m[2][0], v.w * m[3][0]))),
        fma(v.x, m[0][1], fma(v.y, m[1][1], fma(v.z, m[2][1], v.w * m[3][1]))),
        fma(v.x, m[0][2], fma(v.y, m[1][2], fma(v.z, m[2][2], v.w * m[3][2]))),
        fma(v.x, m[0][3], fma(v.y, m[1][3], fma(v.z, m[2][3], v.w * m[3][3]))),
    };
}
SIMD_INLINE v_float3 mul(const glm::mat3& m, const v_float3& n) {
    return {
        fma(n.x, m[0][0], fma(n.y, m[1][0], n.z * m[2][0])),
        fma(n.x, m[0][1], fma(n.y, m[1][1], n.z * m[2][1])),
        fma(n.x, m[0][2], fma(n.y, m[1][2], n.z * m[2][2])),
    };
};

SIMD_INLINE v_float4 perspective_div(const v_float4& v) {
    v_float rw = 1.0f / v.w;
    return { v.x * rw, v.y * rw, v.z * rw, rw };
}

//////////////////////////////////////// Bitwise ////////////////////////////////////////

// https://en.wikipedia.org/wiki/Find_first_set#Properties_and_relations
// https://stackoverflow.com/a/58827596
SIMD_TFN_INT T popcnt(T x) { return __builtin_elementwise_popcount(x); }
SIMD_TFN_INT T lzcnt(T x) { return __builtin_elementwise_clzg(x); }
SIMD_TFN_INT T tzcnt(T x) { return __builtin_elementwise_ctzg(x); }

template<std::integral T>
SIMD_INLINE constexpr uint32_t popcnt(T value) {
    using U = std::make_unsigned_t<T>;
    return (uint32_t)__builtin_popcountg(static_cast<U>(value));
}
template<std::integral T>
SIMD_INLINE constexpr uint32_t tzcnt(T value) {
    using U = std::make_unsigned_t<T>;
    return (uint32_t)__builtin_ctzg(static_cast<U>(value));
}
template<std::integral T>
SIMD_INLINE constexpr uint32_t lzcnt(T value) {
    using U = std::make_unsigned_t<T>;
    return (uint32_t)__builtin_clzg(static_cast<U>(value));
}

// UB for `count` == 0 or 31.
SIMD_INLINE v_uint rotl(v_uint a, int b) { return (a << b) | (a >> (32 - b)); }
SIMD_INLINE v_uint rotr(v_uint a, int b) { return (a >> b) | (a << (32 - b)); }

//////////////////////////////////////// Misc ////////////////////////////////////////

struct AlignedDeleter {
    void operator()(void* data) const { _mm_free(data); }
};

template<typename T>
using AlignedBuffer = std::unique_ptr<T[], AlignedDeleter>;

template<typename T>
AlignedBuffer<T> alloc_buffer(size_t count, size_t align = 64) {
    T* ptr = (T*)_mm_malloc(count * sizeof(T), align);
    return AlignedBuffer<T>(ptr);
}

template<typename T = uint32_t>
class BitIter {
    T mask;

public:
    BitIter(T mask_) : mask(mask_) {}

    BitIter& operator++() {
        mask &= (mask - 1);
        return *this;
    }
    uint32_t operator*() const { return (uint32_t)__builtin_ctzg(mask); }
    friend bool operator!=(const BitIter& a, const BitIter& b) { return a.mask != b.mask; }

    BitIter begin() const { return *this; }
    BitIter end() const { return BitIter(0); }
};

};  // namespace simd
