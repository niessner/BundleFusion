//
//  binary/bitstream.hpp
//  Uplink
//
//  Copyright (c) 2012 Occipital, Inc. All rights reserved.
//

// Occipital bitstream
//
// adapted from code originally written by Orion Hodson
//
// Key changes:
//   - Performance
//   - Use of temporary uint8_t (during write)
// TODOs:
//   - Temporary char should be used during read too  
//

# pragma once

# include "./bitstream.h"
# include <stdint.h>
# include <cstdlib>
# include <cstdio>
# include <cstring>
# include <cassert>

namespace uplink {

//------------------------------------------------------------------------------

inline int
bs_init(bitstream_t *bp)
{
    if (bp) {
        memset(bp, 0, sizeof(bitstream_t));
        return BITSTREAM_SUCCESS;
    }
    return BITSTREAM_FAILURE;
}

inline int
bs_destroy(bitstream_t * b)
{
    free(b);
    return BITSTREAM_SUCCESS;
}

inline int
bs_attach(bitstream_t *b,
          uint8_t *buf,
          int blen)
{
    b->buf    = b->pos = buf;
    b->remain = 8;
    b->len    = blen;
    return BITSTREAM_SUCCESS;
}

inline int
bs_bytes_used(bitstream_t *b)
{
    unsigned int used = (unsigned int)(b->pos - b->buf);
    return b->remain != 8 ? used + 1 : used;
}

inline uint8_t
bs_get(bitstream_t * b,
       uint8_t  nbits)
{
    uint8_t out;
  	
    if (b->remain == 0) {
        b->pos++;
        b->remain = 8;
    }
	
    if (nbits > b->remain) {
        /* Get high bits */
        out = *b->pos;
        out <<= (8 - b->remain);
        out >>= (8 - nbits);
        b->pos++;
        b->remain += 8 - nbits;
        out |= (*b->pos) >> b->remain;
    } else {
        out = *b->pos;
        out <<= (8 - b->remain);
        out >>= (8 - nbits);
        b->remain -= nbits;
    }
    
    assert((unsigned int)(b->pos - b->buf) <= b->len);
    return out;
}

//------------------------------------------------------------------------------

}
