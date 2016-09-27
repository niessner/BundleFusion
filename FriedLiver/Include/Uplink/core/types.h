//
//  core/types.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./config.h"
# include "./macros.h"

# include <stdint.h>
# include <stdarg.h>
# include <memory>
# include <vector>
# include <string>
# include <list>
# include <algorithm>
# include <cmath>
# include <cfloat>
# include <cstdlib>

# if defined(_MSC_VER) && _MSC_VER < 1800
static const unsigned long __nan[2] = {0xffffffff, 0x7fffffff};
#	define NAN (*(const float *) __nan)
#	define isnan _isnan
# endif

namespace uplink {

//------------------------------------------------------------------------------

typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef int8_t  int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

typedef uint8             Byte;
typedef Byte*             Bytes;

typedef std::vector<Byte> Buffer;

typedef std::ptrdiff_t    Index;
typedef std::size_t       Size;

typedef char  const* CString;
typedef char* CMutableString;
typedef std::string   String;

template < typename Type >
void
typed_free (Type* ptr)
{
    std::free(ptr);
}

typedef std::unique_ptr<char, void(*)(char*)> CStringPtr;

CStringPtr formatted      (CString format, ...);
 String    formatted_copy (CString format, ...);

bool operator == (const Buffer& buffer, const String& string);
bool operator == (const String& string, const Buffer& buffer);

typedef std::list<std::string> Strings;

enum Ownership
{
        NoOwnership,
      SelfOwnership,
    SharedOwnership
};

inline void
uplink_swap(Ownership& first, Ownership& second)
{
    ::std::swap(first, second);
}

//------------------------------------------------------------------------------

template < typename Type >
inline String
toString (const Type& instance)
{
    return instance.toString();
}

# define UPLINK_PRINTABLE_TYPE(Type, Format) \
inline String \
toString (const Type& value) \
{ \
    return formatted_copy(Format, value); \
}

UPLINK_PRINTABLE_TYPE(uint8 , "%d")
UPLINK_PRINTABLE_TYPE(uint16, "%d")
UPLINK_PRINTABLE_TYPE(uint32, "%d")
UPLINK_PRINTABLE_TYPE(uint64, "%d")
UPLINK_PRINTABLE_TYPE( int8 , "%d")
UPLINK_PRINTABLE_TYPE( int16, "%d")
UPLINK_PRINTABLE_TYPE( int32, "%d")
UPLINK_PRINTABLE_TYPE( int64, "%d")
UPLINK_PRINTABLE_TYPE(bool  , "%d")
UPLINK_PRINTABLE_TYPE(float , "%f")
UPLINK_PRINTABLE_TYPE(double, "%f")

//------------------------------------------------------------------------------

template < typename T >
inline bool modified (T& lhs, const T& rhs)
{
    const bool ret = lhs != rhs;

    lhs = rhs;

    return ret;
}

template < class Class >
struct Clonable
{
    virtual ~Clonable () {}

    virtual Class* clone () const = 0;
};

    
template < class Derived, class Base >
inline Derived*
downcast (Base* base)
{
#ifndef DEBUG
    return static_cast<Derived*>(base);
#else
    Derived* const derived = dynamic_cast<Derived*>(base);
    assert(0 != derived);
    return derived;
#endif
}

inline String
prettyByteSize (double size)
{
    const char* suffixes [] = { "  ", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB" };
    int n = 0;
    while (size > 1024 && n < sizeof_array(suffixes))
    {
        size /= 1024;
        ++n;
    }
    return formatted_copy("%7.2f %s", size, suffixes[n]);
}

inline String
prettyByteSize (size_t size)
{
    return prettyByteSize(double(size));
}

//------------------------------------------------------------------------------

}

# include "./types.hpp"
