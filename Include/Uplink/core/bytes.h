//
//  binary/bytes.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./types.h"
# include <algorithm>

namespace uplink {

//------------------------------------------------------------------------------

// In the remainder of this file, bytes are numbered using indices of increasing significance.
// As such, 0 is always the least significant byte.

template < unsigned > struct Nth {};

//------------------------------------------------------------------------------

inline Byte getByte (Nth<0>, uint16 val) { return Byte( val & 0x00FF       ); }
inline Byte getByte (Nth<1>, uint16 val) { return Byte((val & 0xFF00) >> 8 ); }

inline Byte getByte (Nth<0>, uint32 val) { return Byte( val & 0x000000FF       ); }
inline Byte getByte (Nth<1>, uint32 val) { return Byte((val & 0x0000FF00) >> 8 ); }
inline Byte getByte (Nth<2>, uint32 val) { return Byte((val & 0x00FF0000) >> 16); }
inline Byte getByte (Nth<3>, uint32 val) { return Byte((val & 0xFF000000) >> 24); }

inline Byte getByte (Nth<0>, uint64 val) { return Byte( val & 0x00000000000000FF       ); }
inline Byte getByte (Nth<1>, uint64 val) { return Byte((val & 0x000000000000FF00) >> 8 ); }
inline Byte getByte (Nth<2>, uint64 val) { return Byte((val & 0x0000000000FF0000) >> 16); }
inline Byte getByte (Nth<3>, uint64 val) { return Byte((val & 0x00000000FF000000) >> 24); }
inline Byte getByte (Nth<4>, uint64 val) { return Byte((val & 0x000000FF00000000) >> 32); }
inline Byte getByte (Nth<5>, uint64 val) { return Byte((val & 0x0000FF0000000000) >> 40); }
inline Byte getByte (Nth<6>, uint64 val) { return Byte((val & 0x00FF000000000000) >> 48); }
inline Byte getByte (Nth<7>, uint64 val) { return Byte((val & 0xFF00000000000000) >> 56); }

template < unsigned Index > inline Byte getByte (uint16 val) { return getByte(Nth<Index>(), val); }
template < unsigned Index > inline Byte getByte (uint32 val) { return getByte(Nth<Index>(), val); }
template < unsigned Index > inline Byte getByte (uint64 val) { return getByte(Nth<Index>(), val); }

//------------------------------------------------------------------------------

inline void setByte (Nth<0>, Byte byte, uint16& val) { val = (val & 0xFF00) |  uint16(byte)       ; }
inline void setByte (Nth<1>, Byte byte, uint16& val) { val = (val & 0x00FF) | (uint16(byte) << 8 ); }

inline void setByte (Nth<0>, Byte byte, uint32& val) { val = (val & 0xFFFFFF00) |  uint32(byte)       ; }
inline void setByte (Nth<1>, Byte byte, uint32& val) { val = (val & 0xFFFF00FF) | (uint32(byte) << 8 ); }
inline void setByte (Nth<2>, Byte byte, uint32& val) { val = (val & 0xFF00FFFF) | (uint32(byte) << 16); }
inline void setByte (Nth<3>, Byte byte, uint32& val) { val = (val & 0x00FFFFFF) | (uint32(byte) << 24); }

inline void setByte (Nth<0>, Byte byte, uint64& val) { val = (val & 0xFFFFFFFFFFFFFF00) |  uint64(byte)       ; }
inline void setByte (Nth<1>, Byte byte, uint64& val) { val = (val & 0xFFFFFFFFFFFF00FF) | (uint64(byte) << 8 ); }
inline void setByte (Nth<2>, Byte byte, uint64& val) { val = (val & 0xFFFFFFFFFF00FFFF) | (uint64(byte) << 16); }
inline void setByte (Nth<3>, Byte byte, uint64& val) { val = (val & 0xFFFFFFFF00FFFFFF) | (uint64(byte) << 24); }
inline void setByte (Nth<4>, Byte byte, uint64& val) { val = (val & 0xFFFFFF00FFFFFFFF) | (uint64(byte) << 32); }
inline void setByte (Nth<5>, Byte byte, uint64& val) { val = (val & 0xFFFF00FFFFFFFFFF) | (uint64(byte) << 40); }
inline void setByte (Nth<6>, Byte byte, uint64& val) { val = (val & 0xFF00FFFFFFFFFFFF) | (uint64(byte) << 48); }
inline void setByte (Nth<7>, Byte byte, uint64& val) { val = (val & 0x00FFFFFFFFFFFFFF) | (uint64(byte) << 56); }

template < unsigned Index > inline void setByte (Byte byte, uint16& val) { setByte(Nth<Index>(), byte, val); }
template < unsigned Index > inline void setByte (Byte byte, uint32& val) { setByte(Nth<Index>(), byte, val); }
template < unsigned Index > inline void setByte (Byte byte, uint64& val) { setByte(Nth<Index>(), byte, val); }

//------------------------------------------------------------------------------

inline uint64 byteOrderSwapped (uint64 val);
inline uint32 byteOrderSwapped (uint32 val);
inline uint16 byteOrderSwapped (uint16 val);

//------------------------------------------------------------------------------

void         appendBuffer      (      Buffer& buffer, const Byte* bytes, Size size);
      Byte*    growBuffer      (      Buffer& buffer, Size increase);
      Byte* mutableBufferBytes (      Buffer& buffer, Index offset = 0);
const Byte*        bufferBytes (const Buffer& buffer, Index offset = 0);

//------------------------------------------------------------------------------

}

#include "bytes.hpp"
