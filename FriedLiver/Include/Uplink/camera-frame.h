//
//  camera-frame.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./image.h"
# include "./camera-calibration.h"
# include "./camera-fixedparams.h"

namespace uplink {

//------------------------------------------------------------------------------

struct CameraFrame : Message
{
    UPLINK_MESSAGE_CLASS(CameraFrame);

    virtual bool serializeWith (Serializer& s)
    {
        if (s.isWriter())
        {
            Writer& w = s.asWriter();

            report_false_unless("Cannot write RGBD frame depth", w.write(depthImage));
            report_false_unless("Cannot write RGBD frame color", w.write(colorImage));
        }
        else
        {
            Reader& r = s.asReader();

            report_false_unless("Cannot read RGBD frame depth", r.read(depthImage));
            report_false_unless("Cannot read RGBD frame color", r.read(colorImage));
        }

        return true;
    }

    void swapWith (CameraFrame& other)
    {
        Message::swapWith(other);

        depthImage.swapWith(other.depthImage);
        colorImage.swapWith(other.colorImage);
    }

    Image depthImage;
    Image colorImage;
};

//------------------------------------------------------------------------------

}

# include "./camera-frame.hpp"
