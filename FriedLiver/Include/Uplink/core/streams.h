//
//  binary/streams.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./types.h"
# include "./bytes.h"
# include <cstring>
# include <fstream>

namespace uplink {

//------------------------------------------------------------------------------

class InputStream
{
public:
    enum { UnknownSize = Size(-1) };

public:
    virtual ~InputStream () {}

public:
    virtual bool read (Byte* bytes, Size size) = 0;
    virtual Size available () const { return UnknownSize; }
};

//------------------------------------------------------------------------------

class OutputStream
{
public:
    virtual ~OutputStream () {}

public:
    virtual bool write (const Byte* bytes, Size size) = 0;
};

//------------------------------------------------------------------------------

struct DuplexStream
: InputStream
, OutputStream
{

};

//------------------------------------------------------------------------------

struct Streamable
{
    virtual bool  readFrom (InputStream & input )       = 0;
    virtual bool writeTo   (OutputStream& output) const = 0;
};

//------------------------------------------------------------------------------

struct RawBufferInputStream : InputStream
{
    RawBufferInputStream (Byte* input, Size length)
        : input(input)
        , length(length)
        , offset(0)
        , count(0)
    {

    }

    virtual bool read (Byte* bytes, Size size)
    {
        if (length < (offset + size))
            return false;

        std::memcpy(bytes, input + offset, size);
        offset += size;
        count  += size;
        return true;
    }

    virtual Size available () const
    {
        return length - offset;
    }

    Byte*  input;
    Size   length;
    Index  offset;
    Index  count ;
};

//------------------------------------------------------------------------------

struct BufferInputStream : InputStream
{
    BufferInputStream (const Buffer& buffer, Index offset = 0)
        : buffer(buffer)
        , offset(offset)
        , count(0)
    {
        assert(size_t(offset) < buffer.size());
    }

    virtual bool read (Byte* bytes, Size size)
    {
        if (buffer.size() < (offset + size))
            return false;

        std::memcpy(bytes, &buffer[offset], size);
        offset += size;
        count  += size;
        return true;
    }

    virtual Size available () const
    {
        return buffer.size() - offset;
    }

    const Buffer& buffer;
    Index         offset;
    Index         count ;
};

//------------------------------------------------------------------------------

struct BufferOutputStream : OutputStream
{
    BufferOutputStream (Buffer& buffer, Index offset = 0)
        : buffer(buffer)
        , offset(offset)
        , count(0)
    {
        assert(size_t(offset) <= buffer.size());
    }

    virtual bool write (const Byte* bytes, Size size)
    {
        appendBuffer(buffer, bytes, size);
        offset += size;
        count  += size;
        return true;
    }

    Buffer& buffer;
    Index   offset;
    Index   count ;
};

//------------------------------------------------------------------------------

class FileInputStream : public InputStream
{
public:
    explicit FileInputStream (CString path);
    virtual ~FileInputStream ();

public:
    virtual bool read (Byte* bytes, Size size);

private:
    std::ifstream input;
};

//------------------------------------------------------------------------------

class FileOutputStream : public OutputStream
{
public:
    explicit FileOutputStream (CString path);
    virtual ~FileOutputStream ();

public:
    virtual bool write (const Byte* bytes, Size size);

private:
    std::ofstream output;
};

//------------------------------------------------------------------------------

struct DuplexFileStream : FileInputStream, FileOutputStream
{
    DuplexFileStream (CString input, CString output);
};

}

#include "./streams.hpp"
