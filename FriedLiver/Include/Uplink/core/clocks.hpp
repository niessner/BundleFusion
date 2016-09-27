//
//  core/clocks.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./clocks.hpp"

namespace uplink {

//------------------------------------------------------------------------------

inline
RateEstimator::RateEstimator (int window)
    : count(0)
    , first(0.)
    , last(0.)
    , window(window)
    , windowed(0.f)
{
    assert(0 < window);
}

inline void
RateEstimator::reset ()
{
    count    = 0;
    first    = 0.;
    last     = 0.;
    windowed = 0.f;
}

inline void
RateEstimator::tick ()
{
    last = double(getTickCount());

    if (count == 0)
    {
        first = last;
    }
    else if (count < window)
    {
        windowed = float(count) / float((last - first) / getTickFrequency());
    }
    else if (count % window == 0)
    {
        windowed = float(window) / float((last - first) / getTickFrequency());

        first = last;
    }

    ++count;
}

inline float
RateEstimator::windowedRate () const
{
    return windowed;
}

inline float
RateEstimator::instantRate () const
{
    return 1.f / float((getTickCount() - last) / getTickFrequency());
}

//------------------------------------------------------------------------------

}
