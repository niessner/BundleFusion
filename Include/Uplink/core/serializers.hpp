//
//  binary/archives.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./serializers.h"
# include "./macros.h"
# include "./config.h"
# include "./bytes.h"
# include <cassert>
# include <cstdarg>
# include <cstring>
# include <limits>

namespace uplink {

//------------------------------------------------------------------------------

inline void
store (uint8 val, Byte*& bytes)
{
    *bytes++ = val;
}

inline void
store (uint16 val, Byte*& bytes)
{
# if UPLINK_SERIALIZERS_USE_HOST_BYTE_ORDER
    *reinterpret_cast<uint16*>(bytes) = val;
    bytes += 2;
# else
    *bytes++ = getByte<1>(val);
    *bytes++ = getByte<0>(val);
# endif
}

inline void
store (uint32 val, Byte*& bytes)
{
# if UPLINK_SERIALIZERS_USE_HOST_BYTE_ORDER
    *reinterpret_cast<uint32*>(bytes) = val;
    bytes += 4;
# else
    *bytes++ = getByte<3>(val);
    *bytes++ = getByte<2>(val);
    *bytes++ = getByte<1>(val);
    *bytes++ = getByte<0>(val);
# endif
}

inline void
store (uint64 val, Byte*& bytes)
{
# if UPLINK_SERIALIZERS_USE_HOST_BYTE_ORDER
    *reinterpret_cast<uint64*>(bytes) = val;
    bytes += 8;
# else
    *bytes++ = getByte<7>(val);
    *bytes++ = getByte<6>(val);
    *bytes++ = getByte<5>(val);
    *bytes++ = getByte<4>(val);
    *bytes++ = getByte<3>(val);
    *bytes++ = getByte<2>(val);
    *bytes++ = getByte<1>(val);
    *bytes++ = getByte<0>(val);
# endif
}

inline void store (int8  val, Byte*& bytes) { store(*reinterpret_cast<const uint8 *>(&val), bytes); }
inline void store (int16 val, Byte*& bytes) { store(*reinterpret_cast<const uint16*>(&val), bytes); }
inline void store (int32 val, Byte*& bytes) { store(*reinterpret_cast<const uint32*>(&val), bytes); }
inline void store (int64 val, Byte*& bytes) { store(*reinterpret_cast<const uint64*>(&val), bytes); }

inline void store (bool   val, Byte*& bytes) { store(*reinterpret_cast<const uint8 *>(&val), bytes); }
inline void store (float  val, Byte*& bytes) { store(*reinterpret_cast<const uint32*>(&val), bytes); }
inline void store (double val, Byte*& bytes) { store(*reinterpret_cast<const uint64*>(&val), bytes); }

//------------------------------------------------------------------------------

inline void
fetch (uint8& val, const Byte*& bytes)
{
    val = *bytes++;
}

inline void
fetch (uint16& val, const Byte*& bytes)
{
# if UPLINK_SERIALIZERS_USE_HOST_BYTE_ORDER
    val = *reinterpret_cast<const uint16*>(bytes);
    bytes += 2;
# else
    setByte<1>(*bytes++, val);
    setByte<0>(*bytes++, val);
# endif
}

inline void
fetch (uint32& val, const Byte*& bytes)
{
# if UPLINK_SERIALIZERS_USE_HOST_BYTE_ORDER
    val = *reinterpret_cast<const uint32*>(bytes);
    bytes += 4;
# else
    setByte<3>(*bytes++, val);
    setByte<2>(*bytes++, val);
    setByte<1>(*bytes++, val);
    setByte<0>(*bytes++, val);
# endif
}

inline void
fetch (uint64& val, const Byte*& bytes)
{
# if UPLINK_SERIALIZERS_USE_HOST_BYTE_ORDER
    val = *reinterpret_cast<const uint64*>(bytes);
    bytes += 8;
# else
    setByte<7>(*bytes++, val);
    setByte<6>(*bytes++, val);
    setByte<5>(*bytes++, val);
    setByte<4>(*bytes++, val);
    setByte<3>(*bytes++, val);
    setByte<2>(*bytes++, val);
    setByte<1>(*bytes++, val);
    setByte<0>(*bytes++, val);
# endif
}

inline void fetch (int8  & val, const Byte*& bytes) { fetch(*reinterpret_cast<uint8 *>(&val), bytes); }
inline void fetch (int16 & val, const Byte*& bytes) { fetch(*reinterpret_cast<uint16*>(&val), bytes); }
inline void fetch (int32 & val, const Byte*& bytes) { fetch(*reinterpret_cast<uint32*>(&val), bytes); }
inline void fetch (int64 & val, const Byte*& bytes) { fetch(*reinterpret_cast<uint64*>(&val), bytes); }

inline void fetch (bool  & val, const Byte*& bytes) { fetch(*reinterpret_cast<uint8 *>(&val), bytes); }
inline void fetch (float & val, const Byte*& bytes) { fetch(*reinterpret_cast<uint32*>(&val), bytes); }
inline void fetch (double& val, const Byte*& bytes) { fetch(*reinterpret_cast<uint64*>(&val), bytes); }

//------------------------------------------------------------------------------

inline Serializer::~Serializer () {}

inline bool
Serializer::put (Serializable& serializable)
{
    return serializable.serializeWith(*this);
}

inline Reader&
Serializer::asReader ()
{
    return *downcast<Reader>(this);
}

inline Writer&
Serializer::asWriter ()
{
    return *downcast<Writer>(this);
}

//------------------------------------------------------------------------------

inline bool
Reader::read (Serializable& serializable)
{
    return serializable.serializeWith(*this);
}

inline bool
Reader::readMagic (CString magic)
{
    const uint8 length = uint8(std::strlen(magic));

    assert(length <= 255);

    char str [256];

    return_false_unless(readBytes(reinterpret_cast<Byte*>(str), length));

    return 0 == std::strncmp(magic, str, length);
}

inline bool
Reader::readString (String& string, uint32 maxLength)
{
    uint32 length;

    return_false_unless(read(length));

    return_false_unless(length <= maxLength);

    if (0 == length)
    {
        string.clear();

        return true;
    }

    string.resize(length, 0);

    Byte* bytes = reinterpret_cast<Byte*>(&string[0]);

    return readBytes(bytes, length);
}

//------------------------------------------------------------------------------

inline
InputStreamReader::InputStreamReader (InputStream& input)
    : owned(false)
    , input(&input)
{
}

inline
InputStreamReader::InputStreamReader (Buffer& buffer)
    : owned(true)
    , input(new BufferInputStream(buffer))
{
}

inline
InputStreamReader::~InputStreamReader ()
{
    if (owned)
        delete input;
}

inline bool
InputStreamReader::readBytes (Byte* bytes, Size size)
{
    assert(0 < size);
    return input->read(bytes, size);
}

inline bool
InputStreamReader::readAll (Buffer& buffer)
{
    const Size inputSize = input->available();

    buffer.clear();

    if (0 == inputSize)
        return true;

    if (inputSize == InputStream::UnknownSize)
    {
        // Exhaust input of unknown size.
        Byte byte;
        while (readBytes(&byte, 1))
            buffer.push_back(byte);

        return true;
    }

    // Exhaust input of known size.
    buffer.resize(inputSize);
    return readBytes(mutableBufferBytes(buffer), inputSize);
}

# define UPLINK_SERIALIZER_ATOM(Name, Type) \
inline bool \
InputStreamReader::read (Type& val) \
{ \
    Byte buf[sizeof(Type)]; \
\
    if (!input->read(buf, sizeof(Type))) \
        return false; \
\
    const Byte* ptr = buf; \
    fetch(val, ptr); \
    return true; \
}
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

//------------------------------------------------------------------------------

inline bool
Writer::write (const Serializable& serializable)
{
    return const_cast<Serializable&>(serializable).serializeWith(*this);
}

inline bool
Writer::writeMagic (CString magic)
{
    const size_t length = std::strlen(magic);

    assert(length <= 255);

    return writeBytes(reinterpret_cast<const Byte*>(magic), length);
}

inline bool
Writer::writeString (const String& string, uint32 maxLength)
{
    const uint32 length = uint32(std::min(
        String::size_type(std::numeric_limits<uint32>::max()),
        string.size()
    ));

    return_false_unless(length < maxLength);

    return_false_unless(write(length));

    if (0 == length)
        return true;

    return writeBytes(reinterpret_cast<const Byte*>(&string[0]), length);
}

//------------------------------------------------------------------------------

inline
OutputStreamWriter::OutputStreamWriter (OutputStream& output_)
    : owned(false)
    , output(&output_)
{
}

inline
OutputStreamWriter::OutputStreamWriter (Buffer& buffer)
    : owned(true)
    , output(new BufferOutputStream(buffer))
{
}

inline
OutputStreamWriter::~OutputStreamWriter ()
{
    if (owned)
        delete output;
}

inline bool
OutputStreamWriter::writeBytes (const Byte* bytes , Size size)
{
    assert(0 < size);
    return output->write(bytes, size);
}

//------------------------------------------------------------------------------

# define UPLINK_SERIALIZER_ATOM(Name, Type) \
inline bool \
OutputStreamWriter::write (const Type& val) \
{ \
    Byte buf [sizeof(Type)]; \
    Byte* ptr = buf; \
    store(val, ptr); \
    return output->write(buf, sizeof(Type)); \
}
         UPLINK_SERIALIZER_ATOMS()
# undef  UPLINK_SERIALIZER_ATOM

//------------------------------------------------------------------------------

inline
Serializable::Serializable ()
{
}

inline
Serializable::~Serializable ()
{
}

inline size_t
Serializable::serializedSize () const
{
    SerializedSizeCalculator calculator;

    if (!const_cast<Serializable&>(*this).serializeWith(calculator))
        return size_t(-1);
    
    return calculator.serializedSize;
}

inline bool
Serializable::readFrom (InputStream& input)
{
    InputStreamReader reader(input);

    return serializeWith(reader);
}

inline bool
Serializable::writeTo (OutputStream& output) const
{
    OutputStreamWriter writer(output);

    return const_cast<Serializable*>(this)->serializeWith(writer);
}

//------------------------------------------------------------------------------

}
