#include "Yson.h"
#include <charconv>
#include <stdexcept>

namespace yson {

static bool IsWhitespace(char ch) { return ch <= 0x20; }
static bool IsPunctuation(char ch) { 
    return ch == '{' || ch == '}' || 
           ch == '[' || ch == ']' || 
           ch == ':' || ch == ',' || ch == '.';
}
static bool IsIdentifierChar(char ch) {
    return (ch >= 'A' && ch <= 'Z') ||
           (ch >= 'a' && ch <= 'z') ||
           (ch >= '0' && ch <= '9') ||
           ch == '_';
}
static const char* GetName(ValueType v) {
    switch (v) {
        case kTypeEnd: return "End";
        case kTypeObject: return "Object";
        case kTypeArray: return "Array";
        case kTypeString: return "String";
        case kTypeNumber: return "Number";
        case kTypeInt: return "Int";
        case kTypeIdentifier: return "Identifier";
        default: return "(unknown)";
    }
}

bool Reader::ReadNext() {
    _lastToken = NextToken();
    Key = "";
    
    if (_lastToken.Type == Token::kEOF) return false;

    if (_lastToken.Type == Token::kComma) _lastToken = NextToken();

    if (_lastToken.Type == '}' || _lastToken.Type == ']') {
        PopState(_lastToken.Type == '}' ? kStateObject : kStateArray);
        Type = kTypeEnd;
        return false;
    }

    if (_currState == kStateObject) {
        if (_lastToken.Type != Token::kIdentifier && _lastToken.Type != Token::kString) {
            ReportError("Expected property name");
        }
        Key = _lastToken.strview();
        if (NextToken().Type != Token::kColon) {
            ReportError("Expected colon after property name");
        }
        _lastToken = NextToken();
    }
    Type = ParseValue();
    return Type != kTypeEnd;
}
ValueType Reader::ParseValue() {
    switch (_lastToken.Type) {
        case '{': {
            PushState(kStateObject);
            return kTypeObject;
        }
        case '[': {
            PushState(kStateArray);
            return kTypeArray;
        }
        case Token::kNumber:
        case Token::kInteger:
        case Token::kString:
        case Token::kIdentifier: {
            return (ValueType)_lastToken.Type;
        }
        default: {
            ReportError("Unexpected token");
            return kTypeEnd;
        }
    }
}

void Reader::ReadExpect(ValueType expType) {
    ReadNext();
    if (Type != expType) {
        char msg[64];
        snprintf(msg, sizeof(msg), "Expected %s, got %s", GetName(expType), GetName(Type));
        ReportError(msg, Pos);
    }
}
void Unescape(std::string_view src, std::string &dest, bool append) {
    if (!append) dest.clear();

    dest.reserve(dest.size() + src.size());
    
    for (size_t startPos = 0; startPos < src.size(); ) {
        auto endPos = src.find('\\', startPos);

        if (startPos < endPos) {
            dest.append(src.substr(startPos, endPos - startPos));
        }
        if (endPos >= src.size() - 1) break;

        switch (src[endPos + 1]) {
            default: dest.append(1, src[endPos + 1]); break;
            case 'n': dest.append("\n"); break;
            case 'r': dest.append("\r"); break;
            case 't': dest.append("\t"); break;
            case 'e': dest.append("\x1B"); break;
            case 'u': {
                int cp = 0xFFFD;

                if (endPos + 6 <= src.size()) {
                    std::from_chars(&src[endPos + 2], &src[endPos + 6], cp, 16);
                }
                // Convert to UTF8
                if (cp < 0x80) {
                    dest.append(1, cp);
                } else if (cp < 0x800) {
                    dest.append(1, 0xC0 | (cp >> 6 & 31));
                    dest.append(1, 0x80 | (cp >> 0 & 63));
                } else {
                    dest.append(1, 0xE0 | (cp >> 12 & 15));
                    dest.append(1, 0x80 | (cp >> 6 & 63));
                    dest.append(1, 0x80 | (cp >> 0 & 63));
                }
                endPos += 4;
                break;
            }
        }
        startPos = endPos + 2;
    }
}

Token Reader::NextToken() {
    char ch;

    // Skip whitespace and comments
    while (true) {
        if (Pos >= Len) return Token::kEOF;
        ch = Input[Pos];

        if (IsWhitespace(ch)) {
            Pos++;
        } else if (ch == '#') {
            while (Pos < Len && Input[Pos++] != '\n');
        } else {
            break;
        }
    }

    _lastTokenPos = Pos;
    size_t startPos = Pos;

    if (IsPunctuation(ch)) {
        Pos++;
        return (Token::TokenType)ch;
    }
    if (ch == '\"' || ch == '\'') {
        return ScanString();
    }
    if ((ch >= '0' && ch <= '9') || (ch == '-' || ch == '+')) {
        return ScanNumber();
    }
    if (IsIdentifierChar(ch)) {
        while (Pos < Len && IsIdentifierChar(Input[Pos])) Pos++;
        return Token(Token::kIdentifier, &Input[startPos], Pos - startPos);
    }
    ReportError("Invalid character", startPos, Pos + 1);
    return Token::kEOF;
}
Token Reader::ScanString() {
    char quote = Input[Pos++];
    size_t startPos = Pos;

    while (Pos < Len) {
        auto endPos = std::string_view(Input, Len).find(quote, Pos);
        Pos = (endPos == std::string::npos) ? Len : endPos + 1;

        if (Input[Pos - 2] != '\\') break;
    }
    return Token(Token::kString, &Input[startPos], Pos - startPos - 1);
}
Token Reader::ScanNumber() {
    Token tok;
    std::from_chars_result res;

    if (Pos + 3 < Len && Input[Pos] == '0' && Input[Pos + 1] == 'x') {
        tok.Type = Token::kInteger;
        res = std::from_chars(&Input[Pos + 2], &Input[Len], tok.NumU, 16);
    } else if (Pos + 3 < Len && Input[Pos] == '0' && Input[Pos + 1] == 'b') {
        tok.Type = Token::kInteger;
        res = std::from_chars(&Input[Pos + 2], &Input[Len], tok.NumU, 2);
    } else {
        tok.Type = Token::kNumber;
        res = std::from_chars(&Input[Pos], &Input[Len], tok.NumF, std::chars_format::general);
    }

    if (res.ec != std::errc()) {
        ReportError("Invalid numeric literal", Pos);
        Pos++;
    } else {
        Pos = (size_t)(res.ptr - Input);
    }
    return tok;
}

void Reader::ReportError(const char* msg, size_t startPos, size_t endPos) {
    if (startPos == ~0ull) startPos = _lastTokenPos;
    if (endPos == ~0ull) endPos = Len;

    uint32_t lineNo = 1, columnNo = 1;
    for (size_t i = 0; i < startPos; i++) {
        if (Input[i] == '\n') {
            lineNo++;
            columnNo = 1;
        } else {
            columnNo++;
        }
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%s (ln %d:%d [%zu])", msg, lineNo, columnNo, startPos);
    throw std::runtime_error(msg);
}

// Writer

void Writer::BeginObject(std::string_view prop) {
    if (!prop.empty()) WriteProp(prop);
    if (_currState == kStateArray) WriteComma();

    PushState(kStateObject);
    Buffer.append("{");
    _needsComma = false;
    _needsNewLine = true;
}
void Writer::BeginArray(std::string_view prop) {
    if (!prop.empty()) WriteProp(prop);
    if (_currState == kStateArray) WriteComma();

    PushState(kStateArray);
    Buffer.append("[");
    _needsComma = false;
    _needsNewLine = true;
}
void Writer::EndObject() {
    PopState(kStateObject);

    _needsComma = false;
    _needsNewLine = true;
    WriteComma();
    Buffer.append("}");
}
void Writer::EndArray() {
    PopState(kStateArray);

    _needsComma = false;
    _needsNewLine = true;
    WriteComma();
    Buffer.append("]");
}

size_t append_num(std::string& dest, auto value, auto... args) {
    size_t pos = dest.size();
    dest.resize(pos + 128);

    char* destPtr = &dest[pos];
    auto res = std::to_chars(destPtr, destPtr + 128, value, args...);
    assert(res.ec == std::errc{});

    size_t size = (size_t)(res.ptr - destPtr);
    dest.resize(pos + size);
    return size;
}

void Writer::WriteInt(int64_t value) {
    if (_currState != kStateObject) WriteComma();
    append_num(Buffer, value);
}

void Writer::WriteUInt(uint64_t value, uint32_t base, uint32_t width) {
    if (_currState != kStateObject) WriteComma();

    if (base == 16) {
        Buffer.append("0x");
    } else if (base == 2) {
        Buffer.append("0b");
    }
    size_t pos = Buffer.size();
    size_t size = append_num(Buffer, value, (int)base);

    if (width > 0 && size < width) {
        Buffer.insert(pos, width - size, '0');
    }
}
void Writer::WriteNum(double value) {
    if (_currState != kStateObject) WriteComma();

    // Check bits directly to determine NaN/Inf under -ffast-math.
    // (probably still a mild UB but it doesn't really matter and gets rid of the warning)
    uint64_t bits = std::bit_cast<uint64_t>(value);
    uint64_t exp = bits >> 52 & 2047;
    uint64_t mant = bits & ((1ull << 52) - 1);

    if (exp != 2047) [[likely]] {
        #pragma float_control(precise, on)
        if (value == (float)value) {
            append_num(Buffer, (float)value);
        } else {
            append_num(Buffer, value);
        }
    } else {
        Buffer.append(mant != 0 ? "NaN" : (value < 0 ? "-Infinity" : "Infinity"));
    }
}

void Writer::WriteStr(std::string_view value) {
    if (_currState != kStateObject) WriteComma();

    Buffer.reserve(Buffer.size() + value.size() + 128);
    Buffer.append("\"");

    for (size_t i = 0; i < value.size(); i++) {
        size_t j = i;
        for (; i < value.size(); i++) {
            uint8_t ch = (uint8_t)value[i];
            if (ch < 0x20 || ch == 0x7F || ch == '"') break;
        }
        if (i != j) {
            Buffer.append(&value[j], i - j);
        }

        if (i < value.size()) {
            char ch = value[i];

            if (ch == '"') {
                Buffer.append("\\\"");
            } else if (ch == '\n') {
                Buffer.append("\\\n"); // intended to be \LF
            } else if (ch == '\r') {
                Buffer.append("\\r");
            } else if (ch == '\t') {
                Buffer.append("\\t");
            } else {
                size_t destPos = Buffer.size();
                Buffer.resize(destPos + 6);
                snprintf(&Buffer[destPos], 7, "\\u%04x", ch);
            }
        }
    }
    Buffer.append("\"");
}

void Writer::WriteProp(std::string_view key, bool quoted) {
    assert(_currState == kStateObject);
    WriteComma();

    if (quoted) {
        WriteStr(key);
    } else {
        Buffer.append(key);
    }
    Buffer.append(": ");
}

void Writer::WriteComma() {
    if (_needsComma) {
        Buffer.append(IndentWidth > 0 || _needsNewLine ? "," : ", ");
    }
    if ((_needsComma || _needsNewLine) && IndentWidth > 0) {
        Buffer.append("\n");
        Buffer.append(_depth * IndentWidth, ' ');
    }
    _needsComma = true;
    _needsNewLine = false;
}

};  // namespace yson
