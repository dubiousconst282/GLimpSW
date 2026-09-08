#pragma once

#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
#include <string_view>
#include <vector>

// YSON is a lean subset of YAML, similar to JSON5 - https://spec.json5.org
// - Only the JSON-y parts of YAML are supported
// - Identifiers can only be ASCII: A-Z a-z 0-9 _
// - Comments start with a '#'
// - Hex and binary integers are allowed: 0xFFFF, 0b0101
// - Leading/trailing dot on numeric literals are not allowed
// - Escaping is only required for delimiting quotes and backlash,
//   all other characters (incl. line breaks) are allowed inside strings.
//     - Escape sequences: \n \r \t \e \uXXXX \X
// - Trailing commas are allowed. Current implementation also ignores
//   missing commas, but reliance on this behavior is disencouraged as to
//   keep some compatibility with JSON5/YAML.
//
// This header implements a small streaming parser, writer, and
// helper macros for serializer definitions.
//
// Sample usage:
// ```cpp
// struct Thing { std::string text; std::vector<double> values; };
// YSON_SERIALIZER_STRUCT_INLINE(Thing, text, values); // use _PROTO and _IMPL for separate decl/impl or template types
//
// auto reader = yson::Reader("{ text: 'hello world', values: [100, 3.14159, 1234e-3], }");
// auto actualThing = reader.Parse<Thing>();
// ```
namespace yson {

struct Token {
    enum TokenType {
        kEOF,
        kLBrace = '{', kRBrace = '}',
        kLBracket = '[', kRBracket = ']',
        kColon = ':', kComma = ',', kDot = '.',
        kIdentifier,
        kString,
        kNumber, kInteger,
    };

    TokenType Type = kEOF;
    uint32_t Len = 0;
    union {
        const char* Str;
        double NumF;
        int64_t NumI;
        uint64_t NumU;
    };

    Token() = default;
    Token(TokenType type) : Type(type) {}
    Token(TokenType type, const char* str, uint32_t len)
        : Type(type), Len(len), Str(str) { }

    std::string_view strview() { return { Str, Len }; } 
};

enum ValueType {
    kTypeEnd = Token::kEOF,
    kTypeObject = Token::kLBrace,
    kTypeArray = Token::kLBracket,
    kTypeString = Token::kString,
    kTypeNumber = Token::kNumber,
    kTypeInt = Token::kInteger,
    kTypeIdentifier = Token::kIdentifier
};

void Unescape(std::string_view src, std::string& dest, bool append = false);

template<typename T>
struct Serializer;

struct Reader {
    const char* Input;
    size_t Pos, Len;

    ValueType Type = kTypeEnd;  // type of current value
    std::string_view Key = "";  // name of current property

    Reader(std::string_view input_) : Input(input_.data()), Pos(0), Len(input_.size()) {}

    // Read next value from current object or array, populating `Type` and `Key` properties.
    bool ReadNext();
    void ReadExpect(ValueType expType);

    // Skip children values of current object or array.
    void Skip() {
        if (Type != kTypeObject && Type != kTypeArray) return;
        int depth = 1;

        while (depth > 0) {
            if (ReadNext()) {
                if (Type == kTypeObject || Type == kTypeArray) depth++;
            } else {
                depth--;
            }
        }
    }

    bool Match(std::string_view name, ValueType type) { return Key == name && Type == type; }
    bool MatchObject(std::string_view name) { return Key == name && Type == kTypeObject; }
    bool MatchArray(std::string_view name) { return Key == name && Type == kTypeArray; }

    int64_t GetInt() {
        if (Type == kTypeInt) return _lastToken.NumI;
        if (Type == kTypeNumber) return (int64_t)_lastToken.NumF;
        return 0;
    }
    double GetNum() {
        if (Type == kTypeInt) return (double)_lastToken.NumI;
        if (Type == kTypeNumber) return _lastToken.NumF;
        return 0;
    }
    bool GetBool() {
        if (Type == kTypeIdentifier) return strcmp(_lastToken.Str, "true") == 0;
        return GetNum() != 0;
    }

    int32_t GetI32() { return (int32_t)GetInt(); }
    uint32_t GetU32() { return (uint32_t)GetInt(); }
    float GetFloat() { return (float)GetNum(); }
    
    // Returns view to current string value as encoded in source, without unescaping.
    std::string_view GetRawString() {
        if (Type == kTypeString || Type == kTypeIdentifier) return _lastToken.strview();
        return "";
    }
    std::string GetString() {
        std::string str;
        GetString(str);
        return str;
    }
    void GetString(std::string& dest, bool append = false) { Unescape(GetRawString(), dest, append); }

    template<typename T>
    T Parse() { T obj = {}; Serializer<T>::Read(*this, obj); return obj; }

    template<typename T>
    void Parse(T& dest) { Serializer<T>::Read(*this, dest); }

    void ReportError(const char* msg, size_t startPos = ~0ull, size_t endPos = ~0ull);

private:
    enum State : uint8_t { kStateUndef, kStateObject, kStateArray };

    Token _lastToken;
    size_t _lastTokenPos;

    uint32_t _depth = 0;
    State _currState = kStateUndef;
    State _parentStates[32] = { };

    void PushState(State newState) {
        if ((_currState != kStateObject && _currState != kStateArray && _currState != kStateUndef) || _depth >= sizeof(_parentStates)) {
            ReportError("Invalid object nesting");
            return;
        }
        _parentStates[_depth++] = _currState;
        _currState = newState;
    }
    void PopState(State expected) {
        if (_currState != expected || _depth == 0) {
            ReportError("Unbalanced object nesting");
            return;
        }
        _currState = _parentStates[--_depth];
    }

    ValueType ParseValue();

    Token NextToken();
    Token ScanString();
    Token ScanNumber();
};

struct Writer {
    std::string Buffer;
    uint32_t IndentWidth = 2;

    void BeginObject(std::string_view prop = "");
    void BeginArray(std::string_view prop = "");
    void EndObject();
    void EndArray();

    void WriteProp(std::string_view key, bool quoted = false);

    void WriteInt(int64_t value);
    void WriteUInt(uint64_t value, uint32_t base = 10, uint32_t width = 0);
    void WriteNum(double value);
    void WriteStr(std::string_view value);

    void WriteInt(std::string_view key, int64_t value) { WriteProp(key); WriteInt(value); }
    void WriteUInt(std::string_view key, uint64_t value, uint32_t base = 10, uint32_t width = 0) { WriteProp(key); WriteUInt(value, base, width); }
    void WriteNum(std::string_view key, double value) { WriteProp(key); WriteNum(value); }
    void WriteStr(std::string_view key, std::string_view value) { WriteProp(key); WriteStr(value); }

    template<typename T>
    void Write(const T& obj) { Serializer<T>::Write(*this, obj); }
    
    template<typename T>
    void Write(std::string_view key, const T& obj) { WriteProp(key); Serializer<T>::Write(*this, obj); }

private:
    enum State : uint8_t { kStateUndef, kStateObject, kStateArray };
    
    uint32_t _depth = 0;
    bool _needsComma = false;
    bool _needsNewLine = false;
    State _currState = kStateUndef;
    State _parentStates[32] = { };

    void PushState(State newState) {
        assert(_currState == kStateArray || _currState == kStateObject || _currState == kStateUndef);
        _parentStates[_depth++] = _currState;
        _currState = newState;
    }
    void PopState(State expected) {
        assert(_currState == expected);
        assert(_depth > 0);
        _currState = _parentStates[--_depth];
    }
    void WriteComma();
};

};  // namespace yson

// Serializers

namespace yson {
template<>
struct Serializer<std::string> {
    static void Read(Reader& rd, std::string& str) { rd.GetString(str); }
    static void Write(Writer& wr, const std::string& str) { wr.WriteStr(str); }
};
template<>
struct Serializer<std::string_view> {
    static void Read(Reader& rd, std::string_view& str) { str = rd.GetRawString(); }
    static void Write(Writer& wr, const std::string_view& str) { wr.WriteStr(str); }
};

template<std::integral T>
struct Serializer<T> {
    static void Read(Reader& rd, T& val) { val = T(rd.GetInt()); }
    static void Write(Writer& wr, const T& val) {
        if constexpr (std::is_signed_v<T>) {
            wr.WriteInt(int64_t(val));
        } else {
            wr.WriteUInt(uint64_t(val));
        }
    }
};
template<std::floating_point T>
struct Serializer<T> {
    static void Read(Reader& rd, T& val) { val = T(rd.GetNum()); }
    static void Write(Writer& wr, const T& val) { wr.WriteNum(double(val)); }
};

template<typename E>
struct Serializer<std::vector<E>> {
    static void Read(Reader& rd, std::vector<E>& obj) {
        while (rd.ReadNext()) {
            yson::Serializer<E>::Read(rd, obj.emplace_back());
        }
    }
    static void Write(Writer& wr, const std::vector<E>& obj) {
        wr.BeginArray();
        for (auto& elem : obj) {
            yson::Serializer<E>::Write(wr, elem);
        }
        wr.EndArray();
    }
};

// Default enum serializer maps to int
template<typename T> requires(std::is_enum_v<T>)
struct Serializer<T> {
    static void Read(Reader& rd, T& val) { val = T(rd.GetInt()); }
    static void Write(Writer& wr, const T& val) { wr.WriteUInt(uint64_t(val)); }
};

};  // namespace yson

// Macros

#define YSON_WRITER_FN(type) void yson::Serializer<type>::Write(yson::Writer& wr, const type& obj)
#define YSON_READER_FN(type) void yson::Serializer<type>::Read(yson::Reader& rd, type& obj)

#define YSON_FOR1(op, x) op(x)
#define YSON_FOR2(op, x, ...) op(x) YSON_FOR1(op, __VA_ARGS__)
#define YSON_FOR3(op, x, ...) op(x) YSON_FOR2(op, __VA_ARGS__)
#define YSON_FOR4(op, x, ...) op(x) YSON_FOR3(op, __VA_ARGS__)
#define YSON_FOR5(op, x, ...) op(x) YSON_FOR4(op, __VA_ARGS__)
#define YSON_FOR6(op, x, ...) op(x) YSON_FOR5(op, __VA_ARGS__)
#define YSON_FOR7(op, x, ...) op(x) YSON_FOR6(op, __VA_ARGS__)
#define YSON_FOR8(op, x, ...) op(x) YSON_FOR7(op, __VA_ARGS__)
#define YSON_FOR9(op, x, ...) op(x) YSON_FOR8(op, __VA_ARGS__)
#define YSON_FOR10(op, x, ...) op(x) YSON_FOR9(op, __VA_ARGS__)
#define YSON_FOR11(op, x, ...) op(x) YSON_FOR10(op, __VA_ARGS__)
#define YSON_FOR12(op, x, ...) op(x) YSON_FOR11(op, __VA_ARGS__)
#define YSON_FOR13(op, x, ...) op(x) YSON_FOR12(op, __VA_ARGS__)
#define YSON_FOR14(op, x, ...) op(x) YSON_FOR13(op, __VA_ARGS__)
#define YSON_FOR15(op, x, ...) op(x) YSON_FOR14(op, __VA_ARGS__)
#define YSON_FOR16(op, x, ...) op(x) YSON_FOR15(op, __VA_ARGS__)
#define YSON_FOR17(op, x, ...) op(x) YSON_FOR16(op, __VA_ARGS__)
#define YSON_FOR18(op, x, ...) op(x) YSON_FOR17(op, __VA_ARGS__)
#define YSON_FOR19(op, x, ...) op(x) YSON_FOR18(op, __VA_ARGS__)
#define YSON_FOR20(op, x, ...) op(x) YSON_FOR19(op, __VA_ARGS__)

#define YSON_FOR_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, N, ...) YSON_FOR##N
#define YSON_FOR_EACH(op, ...) YSON_FOR_N(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)(op, __VA_ARGS__)

#define YSON_SERIALIZER_PROTO(type, ...)                      \
    template<__VA_ARGS__>                                     \
    struct yson::Serializer<type> {                           \
        static void Read(yson::Reader& rd, type& obj);        \
        static void Write(yson::Writer& wr, const type& obj); \
    };
    
#define YSON_READ_FIELD__(fld) if (rd.Key == #fld) { yson::Serializer<decltype(obj.fld)>::Read(rd, obj.fld); continue; }
#define YSON_WRITE_FIELD__(fld) wr.WriteProp(#fld); yson::Serializer<decltype(obj.fld)>::Write(wr, obj.fld);

#define YSON_SERIALIZER_STRUCT_IMPL(type, ...)            \
    YSON_READER_FN(type) {                                \
        while (rd.ReadNext()) {                           \
            YSON_FOR_EACH(YSON_READ_FIELD__, __VA_ARGS__) \
        }                                                 \
    }                                                     \
    YSON_WRITER_FN(type) {                                \
        wr.BeginObject();                                 \
        YSON_FOR_EACH(YSON_WRITE_FIELD__, __VA_ARGS__)    \
        wr.EndObject();                                   \
    }

#define YSON_SERIALIZER_STRUCT_INLINE(type, ...)          \
    YSON_SERIALIZER_PROTO(type)                           \
    inline YSON_READER_FN(type) {                         \
        while (rd.ReadNext()) {                           \
            YSON_FOR_EACH(YSON_READ_FIELD__, __VA_ARGS__) \
            rd.Skip();                                    \
        }                                                 \
    }                                                     \
    inline YSON_WRITER_FN(type) {                         \
        wr.BeginObject();                                 \
        YSON_FOR_EACH(YSON_WRITE_FIELD__, __VA_ARGS__)    \
        wr.EndObject();                                   \
    }

#define YSON_READ_ENUM__(name) else if (str == #name) obj = E::name;
#define YSON_WRITE_ENUM__(name) case E::name: str = #name; break;

#define YSON_SERIALIZER_STR_ENUM(type, ...)                     \
    YSON_SERIALIZER_PROTO(type)                                 \
    inline YSON_READER_FN(type) {                               \
        using E = type;                                         \
        auto str = rd.GetRawString();                           \
        if (false)                                              \
            ;                                                   \
        YSON_FOR_EACH(YSON_READ_ENUM__, __VA_ARGS__)            \
        else rd.ReportError("Unknown mapping for enum " #type); \
    }                                                           \
    inline YSON_WRITER_FN(type) {                               \
        using E = type;                                         \
        const char* str = "";                                   \
        switch (obj) {                                          \
            YSON_FOR_EACH(YSON_WRITE_ENUM__, __VA_ARGS__)       \
            default: assert(!"Invalid value for enum " #type);  \
        }                                                       \
        wr.WriteStr(str);                                       \
    }

#ifdef YSON_ENABLE_EXTRA_STD_UNORDERED_MAP
#include <unordered_map>

template<typename V>
struct yson::Serializer<std::unordered_map<std::string, V>> {
    static void Read(Reader& rd, std::unordered_map<std::string, V>& obj) {
        std::string key;
        while (rd.ReadNext()) {
            yson::Unescape(rd.Key, key);
            Serializer<V>::Read(rd, obj[key]);
            key.clear();
        }
    }
    static void Write(Writer& wr, const std::unordered_map<std::string, V>& obj) {
        wr.BeginObject();
        for (auto& [k, v] : obj) {
            wr.WriteProp(k, true);
            Serializer<V>::Write(wr, v);
        }
        wr.EndObject();
    }
};

template<std::integral K, typename V>
struct yson::Serializer<std::unordered_map<K, V>> {
    static void Read(Reader& rd, std::unordered_map<K, V>& obj) {
        while (rd.ReadNext()) {
            K key;
            char* parsedEnd;

            if constexpr (std::is_unsigned_v<K>) {
                key = static_cast<K>(strtoull(rd.Key.data(), &parsedEnd, 10));
            } else {
                key = static_cast<K>(strtoll(rd.Key.data(), &parsedEnd, 10));
            }
            if (parsedEnd != rd.Key.data() + rd.Key.size()) {
                rd.ReportError("Invalid integer key");
                return;
            }
            Serializer<V>::Read(rd, obj[key]);
        }
    }
    static void Write(Writer& wr, const std::unordered_map<K, V>& obj) {
        wr.BeginObject();
        for (auto& [k, v] : obj) {
            wr.WriteProp(std::to_string(k), true);
            Serializer<V>::Write(wr, v);
        }
        wr.EndObject();
    }
};
#endif

#ifdef YSON_ENABLE_EXTRA_GLM
#include <glm/fwd.hpp>

template<typename T, int L>
struct yson::Serializer<glm::vec<L, T>> {
    static void Read(Reader& rd, glm::vec<L, T>& v) {
        for (int i = 0; i < L; i++) {
            rd.ReadExpect(yson::kTypeNumber);
            v[i] = T(rd.GetNum());
        }
        if (rd.ReadNext()) rd.ReportError("Invalid data found after parsing vector");
    }
    static void Write(Writer& wr, const glm::vec<L, T>& v) {
        uint32_t oldIndent = wr.IndentWidth;
        wr.IndentWidth = 0;
        wr.BeginArray();
        for (int i = 0; i < L; i++) wr.WriteNum(double(v[i]));
        wr.EndArray();
        wr.IndentWidth = oldIndent;
    }
};

template<typename T, int R, int C>
struct yson::Serializer<glm::mat<C, R, T>> {
    static void Read(Reader& rd, glm::mat<C, R, T>& m) {
        for (int i = 0; i < C * R; i++) {
            rd.ReadExpect(yson::kTypeNumber);
            m[i / R][i % R] = T(rd.GetNum());
        }
        if (rd.ReadNext()) rd.ReportError("Invalid data found after parsing matrix");
    }
    static void Write(Writer& wr, const glm::mat<C, R, T>& m) {
        uint32_t oldIndent = wr.IndentWidth;
        wr.BeginArray();
        for (int ci = 0; ci < C; ci++) {
            for (int ri = 0; ri < R; ri++) {
                wr.IndentWidth = ri == 0 ? oldIndent : 0;
                wr.WriteNum(double(m[ci][ri]));
            }
        }
        wr.IndentWidth = oldIndent;
        wr.EndArray();
    }
};
#endif