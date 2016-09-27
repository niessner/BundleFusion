//
//  core/types.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./types.h"

# if UPLINK_HAS_CXX11 || _MSC_VER >= 1600 || (__APPLE__ && !__GLIBCXX__)
#   include <memory>
#   define uplink_ref ::std::shared_ptr
# else
#   include <tr1/memory>
#   define uplink_ref ::std::tr1::shared_ptr
# endif

namespace uplink {

int vasprintf (char** output, const char* format, va_list args);

//------------------------------------------------------------------------------

// Somebody explain me what's going on with std::swap and C++11.
//
// This code breaks std::vector::swap instantiations with clang & libc++:
//
// template < typename Type >
// void swap (Type& first, Type& second)
// {}
//
// So, for now, let's use: 'uplink_swap(a,b)' and 'a.swapWith(b)'.

# define UPLINK_SWAPPABLE_TYPE(Type) \
inline void \
uplink_swap (Type& first, Type& second) \
{ \
    ::std::swap(first, second); \
}

UPLINK_SWAPPABLE_TYPE(void*)
UPLINK_SWAPPABLE_TYPE(bool)
UPLINK_SWAPPABLE_TYPE(uint8)
UPLINK_SWAPPABLE_TYPE(uint16)
UPLINK_SWAPPABLE_TYPE(uint32)
UPLINK_SWAPPABLE_TYPE(uint64)
UPLINK_SWAPPABLE_TYPE(int8)
UPLINK_SWAPPABLE_TYPE(int16)
UPLINK_SWAPPABLE_TYPE(int32)
UPLINK_SWAPPABLE_TYPE(int64)
UPLINK_SWAPPABLE_TYPE(float)
UPLINK_SWAPPABLE_TYPE(double)
UPLINK_SWAPPABLE_TYPE(std::string)
#if __APPLE__
UPLINK_SWAPPABLE_TYPE(size_t)
#endif

# undef UPLINK_SWAPPABLE_TYPE

template < class Class >
inline void
uplink_swap (Class*& first, Class*& second)
{
    std::swap(first, second);
}

template < class Class >
inline void
uplink_swap (const Class*& first, const Class*& second)
{
    std::swap(first, second);
}

template < class Class >
inline void
uplink_swap (uplink_ref<Class>& first, uplink_ref<Class>& second)
{
    first.swap(second);
}

template < class Class >
inline void
uplink_swap (std::vector<Class>& first, std::vector<Class>& second)
{
    first.swap(second);
}

template < class Class >
inline void
uplink_swap (Class& first, Class& second)
{
    first.swapWith(second);
}

inline CStringPtr
formatted (CString format, ...)
{
    char* buf;
    va_list args;
    va_start(args, format);
    vasprintf(&buf, format, args);
    va_end(args);
    return CStringPtr(buf, typed_free<char>);
}

inline String
formatted_copy (CString format, ...)
{
    char* buf;
    va_list args;
    va_start(args, format);
    vasprintf(&buf, format, args);
    va_end(args);
    CStringPtr freed(buf, typed_free<char>);
    return String(freed.get());
}

inline bool
operator == (const Buffer& buffer, const String& string)
{
    return std::equal(buffer.begin(), buffer.end(), string.begin());
}

inline bool
operator == (const String& string, const Buffer& buffer)
{
    return buffer == string;
}

//------------------------------------------------------------------------------

}
