//
//  core/clocks.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./config.h"

namespace uplink {

//------------------------------------------------------------------------------

inline double
getTime ()
{
    return double(getTickCount()) / double(getTickFrequency());
}

static const double never = std::numeric_limits<double>::lowest();

//------------------------------------------------------------------------------

class RateEstimator
{
public:
    enum { DefaultWindow = 10 };

public:
    explicit RateEstimator (int window  = DefaultWindow);

public:
    void reset ();
    void tick ();

public:
    float windowedRate () const;
    float  instantRate () const;

public:
    String toString () const
    {
        return formatted_copy("%7.2f Hz", windowedRate());
    }

private:
    int    count;
    double first;
    double last;
    int    window;
    float  windowed;
};

//------------------------------------------------------------------------------

class StopWatch
{
public:
    StopWatch ()
    : startTime(0.)
    {
    }

public:
    void start ()
    {
        startTime = getTime();
    }
    
    double elapsed () const
    {
        return getTime() - startTime;
    }
    
private:
    double startTime;
};

//------------------------------------------------------------------------------

}

# include "./clocks.hpp"
