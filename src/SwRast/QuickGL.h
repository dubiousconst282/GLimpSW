#pragma once

#include <cstdint>
#include <vector>
#include <glad/glad.h>

// Basic OpenGL object wrappers
namespace ogl {

struct Texture2D {
    GLuint Handle;
    uint32_t Width, Height, MipLevels;

    Texture2D(uint32_t width, uint32_t height, uint32_t mipLevels, GLuint fmt) {
        glCreateTextures(GL_TEXTURE_2D, 1, &Handle);
        Width = width;
        Height = height;
        MipLevels = mipLevels;

        glTextureStorage2D(Handle, mipLevels, fmt, width, height);

        glTextureParameteri(Handle, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTextureParameteri(Handle, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glTextureParameteri(Handle, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(Handle, GL_TEXTURE_WRAP_T, GL_REPEAT);
    }
    ~Texture2D() { glDeleteTextures(1, &Handle); }

    void SetPixels(uint32_t* pixels, uint32_t stride) {
        glPixelStorei(GL_UNPACK_ROW_LENGTH, stride);

        glTextureSubImage2D(Handle, 0, 0, 0, Width, Height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glGenerateTextureMipmap(Handle);

        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    }
};

struct VertexBuffer {
    enum IndexFormat { U8 = 1, U16 = 2, U32 = 4 };

    uint32_t VertexCount, IndexCount, VertexStride;
    IndexFormat IndexFmt;

    GLuint VboHandle, EboHandle;

    VertexBuffer(uint32_t vertexCount, uint32_t indexCount, uint32_t vertexStride, IndexFormat indexFmt) {
        VertexCount = vertexCount;
        IndexCount = indexCount;
        VertexStride = vertexCount;
        IndexFmt = indexFmt;

        EboHandle = 0;  // kill warning
        glCreateBuffers(2, &VboHandle);

        glNamedBufferStorage(VboHandle, (size_t)vertexCount * vertexStride, 0, GL_MAP_WRITE_BIT);
        glNamedBufferStorage(EboHandle, (size_t)indexCount * (size_t)indexFmt, 0, GL_MAP_WRITE_BIT);
    }

    ~VertexBuffer() { glDeleteBuffers(2, &VboHandle); }

    void Map(void** vertexBuffer, void** indexBuffer) {
        *vertexBuffer = glMapNamedBuffer(VboHandle, GL_WRITE_ONLY);
        *indexBuffer = glMapNamedBuffer(EboHandle, GL_WRITE_ONLY);
    }
    void Unmap() {
        glUnmapNamedBuffer(VboHandle);
        glUnmapNamedBuffer(EboHandle);
    }
};

struct VertexLayout {
    enum DataType { F32, UNorm8, SNorm8, UNorm16, SNorm16 };
    struct AttribDesc {
        uint16_t Offset, Count;
        DataType Type;
        char Name[64];
    };

    std::vector<AttribDesc> Attribs;
    uint32_t Stride;
};

struct Shader {
private:
    static const uint32_t kMaxTextures = 16;

    GLuint _boundTextures[kMaxTextures]{ 0 };
    GLint _textureUnits[kMaxTextures]{ 0 }; //[uniform loc] -> unitId
    uint32_t _numBoundTextures;
    uint32_t _vertexStride;
    
    GLuint _vaoHandle;

public:
    GLuint Handle;

    Shader() {
        Handle = glCreateProgram();
        glCreateVertexArrays(1, &_vaoHandle);
    }
    ~Shader() {
        if (Handle != 0) {
            glDeleteProgram(Handle);
            Handle = 0;
        }
        if (_vaoHandle != 0) {
            glDeleteVertexArrays(1, &_vaoHandle);
            _vaoHandle = 0;
        }
    }

    void DrawTriangles(const VertexBuffer& vbo, size_t vboOffset, size_t eboOffset, uint32_t count) {
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        glUseProgram(Handle);

        glBindVertexArray(_vaoHandle);
        glBindTextures(0, _numBoundTextures, _boundTextures);

        glVertexArrayVertexBuffer(_vaoHandle, 0, vbo.VboHandle, vboOffset * _vertexStride, _vertexStride);
        glVertexArrayElementBuffer(_vaoHandle, vbo.EboHandle);
        glDrawElements(GL_TRIANGLES, count, GetActualFmt(vbo.IndexFmt), (void*)(eboOffset * (uint32_t)vbo.IndexFmt));
    }

    void SetUniform(std::string_view name, const Texture2D* tex) {
        GLint loc = glGetUniformLocation(Handle, name.data());
        if (loc < 0) return;

        int32_t unit = -1;

        for (int32_t i = 0; i < kMaxTextures; i++) {
            if (_textureUnits[i] == loc) {
                unit = i;
                break;
            }
        }
        if (unit < 0) {
            unit = (int32_t)_numBoundTextures++;
            _textureUnits[unit] = loc;
            glProgramUniform1i(Handle, loc, unit);
        }
        _boundTextures[unit] = tex ? dynamic_cast<const Texture2D*>(tex)->Handle : 0;
    }
    void SetUniform(std::string_view name, const float* values, uint32_t count) {
        GLint loc = glGetUniformLocation(Handle, name.data());

        switch (count) {
            case 1: glProgramUniform1f(Handle, loc, *values); break;
            case 2: glProgramUniform2fv(Handle, loc, 1, values); break;
            case 3: glProgramUniform3fv(Handle, loc, 1, values); break;
            case 4: glProgramUniform4fv(Handle, loc, 1, values); break;
            case 16: glProgramUniformMatrix4fv(Handle, loc, 1, false, values); break;
            default: throw std::exception();
        }
    }

    void Attach(GLuint type, std::string_view source) {
        GLuint shaderId = glCreateShader(type);

        const char* pSource = source.data();
        glShaderSource(shaderId, 1, &pSource, nullptr);

        glCompileShader(shaderId);

        GLint status;
        glGetShaderiv(shaderId, GL_COMPILE_STATUS, &status);

        if (status != GL_TRUE) {
            GLchar infoStr[1024];
            GLsizei infoLen;
            glGetShaderInfoLog(shaderId, sizeof(infoStr), &infoLen, infoStr);

            throw std::exception("Failed to attach shader to program");
        }
        glAttachShader(Handle, shaderId);
    }

    void Link() {
        glLinkProgram(Handle);
        DeleteAttachedShaders();

        GLint status;
        glGetProgramiv(Handle, GL_LINK_STATUS, &status);

        if (status != GL_TRUE) {
            GLchar infoStr[1024];
            GLsizei infoLen;
            glGetProgramInfoLog(Handle, sizeof(infoStr), &infoLen, infoStr);

            throw std::exception("Failed to link shader program");
        }
    }

    void DeleteAttachedShaders() {
        GLuint shaders[16];
        GLsizei count;

        glGetAttachedShaders(Handle, 16, &count, shaders);

        for (GLsizei i = 0; i < count; i++) {
            glDetachShader(Handle, shaders[i]);
            glDeleteShader(shaders[i]);
        }
    }

    void SetVertexLayout(const VertexLayout& layout) {
        for (auto& attrib : layout.Attribs) {
            GLint location = glGetAttribLocation(Handle, attrib.Name);

            if (location < 0) continue;

            glEnableVertexArrayAttrib(_vaoHandle, location);
            glVertexArrayAttribBinding(_vaoHandle, location, 0);

            if (attrib.Type == VertexLayout::F32) {
                glVertexArrayAttribFormat(_vaoHandle, location, attrib.Count, GL_FLOAT, true, attrib.Offset);
            } else {
                glVertexArrayAttribIFormat(_vaoHandle, location, attrib.Count, GetAttribType(attrib.Type), attrib.Offset);
            }
        }
        _vertexStride = layout.Stride;
    }

    GLenum GetActualFmt(VertexBuffer::IndexFormat fmt) {
        switch (fmt) {
            case VertexBuffer::U8: return GL_UNSIGNED_BYTE;
            case VertexBuffer::U16: return GL_UNSIGNED_SHORT;
            case VertexBuffer::U32: return GL_UNSIGNED_INT;
            default: throw std::exception();
        }
    }
    GLenum GetAttribType(VertexLayout::DataType type) {
        switch (type) {
            case VertexLayout::F32: return GL_FLOAT;
            case VertexLayout::UNorm8: return GL_UNSIGNED_BYTE;
            case VertexLayout::SNorm8: return GL_BYTE;
            case VertexLayout::UNorm16: return GL_UNSIGNED_SHORT;
            case VertexLayout::SNorm16: return GL_SHORT;
            default: throw std::exception();
        }
    }
};

}; // namespace ogl