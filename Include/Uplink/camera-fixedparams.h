//
//  camera/camera-fixedparams.h
//  Uplink
//
//  Copyright (c) 2013 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./core/types.h"
# include "./core/serializers.h"

namespace uplink {

//------------------------------------------------------------------------------

class CameraFixedParams : public Message
{
public:
    UPLINK_MESSAGE_CLASS(CameraFixedParams);

public:
    CameraFixedParams ();
    
public:
    void swapWith (CameraFixedParams& other);
    
public:
    bool isValid () const;

public:
    bool serializeWith (Serializer& serializer);

public:
    float CMOSAndEmitterDistance;
    float refPlaneDistance;
    float planePixelSize;
};

//------------------------------------------------------------------------------

}

# include "./camera-fixedparams.hpp"
