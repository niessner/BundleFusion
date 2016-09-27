//
//  bitstream.h
//  Uplink
//
//  Copyright (c) 2012 Occipital, Inc. All rights reserved.
//

# pragma once

# include <stdint.h>
# include <cstdio>
# include <cassert>

# define BITSTREAM_SUCCESS 1
# define BITSTREAM_FAILURE 0

namespace uplink {

//------------------------------------------------------------------------------

typedef struct s_bitstream {
    uint8_t *buf;               /* head of bitstream            */
    uint8_t *pos;               /* current byte in bitstream    */
    unsigned int   remain;      /* bits remaining               */
    unsigned int   len;         /* length of bitstream in bytes */
    uint8_t cur_bits;
} bitstream_t;

int bs_init(bitstream_t *bp);
int bs_destroy(bitstream_t **ppb);
int bs_attach(bitstream_t *b, uint8_t *buf, int blen);
uint8_t bs_get(bitstream_t * b, uint8_t  nbits);
int bs_bytes_used(bitstream_t *b);

static inline int
bs_put(bitstream_t * b,
       uint8_t       bits,
       uint8_t       nbits)
{
    assert(nbits != 0 && nbits <= 8);
    
    if ( nbits > b->remain ) {
        unsigned int over = nbits - b->remain;
        *(b->pos++) = b->cur_bits | (bits >> over);
        b->remain = 8 - over;
        b->cur_bits = bits << b->remain;
    } else {
        b->cur_bits |= bits << (b->remain - nbits);
        b->remain -= nbits;
        
        // we've exhausted the byte.  move to the next byte.
        if (b->remain == 0) {
            *(b->pos++) = b->cur_bits;
            b->remain = 8;
            b->cur_bits = 0;
        }
    }
    
    assert((unsigned int)(b->pos - b->buf) <= b->len);
    return BITSTREAM_SUCCESS;
}

static inline int
bs_flush(bitstream_t * b)
{
    *(b->pos) = b->cur_bits;
    return BITSTREAM_SUCCESS;
}

//------------------------------------------------------------------------------

}

# include "bitstream.hpp"
