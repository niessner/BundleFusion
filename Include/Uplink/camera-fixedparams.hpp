//
//  camera/camera-fixedparams.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./camera-fixedparams.h"
# include "./core/macros.h"

namespace uplink {

//------------------------------------------------------------------------------

inline
CameraFixedParams::CameraFixedParams ()
    : CMOSAndEmitterDistance(-1)
    , refPlaneDistance(-1)
    , planePixelSize(-1)
{}

inline bool
CameraFixedParams::isValid () const
{
    return CMOSAndEmitterDistance > 0.f;
}

inline bool
CameraFixedParams::serializeWith (Serializer& s)
{
# define MEMBER(Name) \
    report_false_unless("cannot archive camera fixed params " #Name, s.put(Name))

    MEMBER(CMOSAndEmitterDistance);
    MEMBER(refPlaneDistance);
    MEMBER(planePixelSize);

# undef MEMBER

    return true;
}
    
inline void
CameraFixedParams::swapWith (CameraFixedParams& other)
{
    Message::swapWith(other);

# define MEMBER(Name) \
uplink_swap(Name, other.Name);
    
    MEMBER(CMOSAndEmitterDistance);
    MEMBER(refPlaneDistance);
    MEMBER(planePixelSize);
    
# undef  MEMBER
}

//------------------------------------------------------------------------------

}
