//
//  binary/streams.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./streams.h"
# include <cassert>

namespace uplink {

//------------------------------------------------------------------------------

inline
FileInputStream::FileInputStream (CString path)
: input(path)
{
}

inline
FileInputStream::~FileInputStream ()
{
}

inline bool
FileInputStream::read (Byte* bytes, Size size)
{
    do
    {
        if (!input || !input.read(reinterpret_cast<CMutableString>(bytes), size))
            return false;

        const int64 num = input.gcount();

        size  -= num;
        bytes += num;
    }
    while (0 < size);

    return true;
}

//------------------------------------------------------------------------------

inline
FileOutputStream::FileOutputStream (CString path)
: output(path)
{
}

inline
FileOutputStream::~FileOutputStream ()
{
}

inline bool
FileOutputStream::write (const Byte* bytes, Size size)
{
    if (!output || !output.write(reinterpret_cast<CString>(bytes), size))
        return false;

    bytes += size;
    return true;
}

//------------------------------------------------------------------------------

inline
DuplexFileStream::DuplexFileStream (CString input, CString output)
    : FileInputStream(input)
    , FileOutputStream(output)
{

}

//------------------------------------------------------------------------------

}
