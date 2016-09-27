//
//  network/sessions-setup.hpp
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./sessions-setup.h"

namespace uplink {

//------------------------------------------------------------------------------

inline
AnySessionSetupAction::
AnySessionSetupAction ()
: kind(SessionSetupActionKind_Invalid)
{

}

inline
AnySessionSetupAction::
AnySessionSetupAction (const AnySessionSetupAction& copy)
: kind(copy.kind)
, that(copy.kind == SessionSetupActionKind_Invalid ? 0 : copy.that->clone())
{}

inline void
AnySessionSetupAction::
swapWith (AnySessionSetupAction& other)
{
    uplink_swap(kind, other.kind);
    uplink_swap(that, other.that);
}

inline void
AnySessionSetupAction::
clear ()
{
    kind = SessionSetupActionKind_Invalid;
    that.reset();
}

inline bool
AnySessionSetupAction::
serializeWith (Serializer& serializer)
{
    if (serializer.isReader())
    {
        that.reset();
        return_false_unless(serializer.asReader().read(kind));
        return_false_if(SessionSetupActionKind_Invalid == kind); // Reading an invalid action doesn't make sense.
        that.reset(newSessionSetupAction(kind));
    }
    else
    {
        return_false_if(SessionSetupActionKind_Invalid == kind); // Writing an invalid action doesn't make sense.
        return_false_unless(serializer.asWriter().write(kind));
    }

    return serializer.put(*that);
}

inline bool
AnySessionSetupAction::
applyTo (SessionSettings& sessionSettings) const
{
    return_false_if(!that); // An invalid action cannot be executed.

    return that->applyTo(sessionSettings);
}

//------------------------------------------------------------------------------

inline
SessionSetup::
SessionSetup ()
{
    // Start with an empty action list.
}

inline void
SessionSetup::
swapWith (SessionSetup& other)
{
    Message::swapWith(other);
    
    uplink_swap(actions, other.actions);
}

inline bool
SessionSetup::
serializeWith (Serializer &serializer)
{
    if(serializer.isReader())
    {
        uint16 size;
        return_false_unless(serializer.asReader().read(size));
        // FIXME: Maximum size check.
        actions.resize(size);
    }
    else
    {
        return_false_unless(serializer.asWriter().write(uint16(actions.size())));
    }

    for (int n = 0; n < actions.size(); ++n)
        return_false_unless(actions[n].serializeWith(serializer));

    return true;
}

inline bool
SessionSetup::
applyTo (SessionSettings& sessionSettings) const
{
    for (int n = 0; n < actions.size(); ++n)
        return_false_unless(actions[n].applyTo(sessionSettings));
    
    return true;
}

inline SessionSetupAction*
newSessionSetupAction (SessionSetupActionKind kind)
{
    switch (kind.value)
    {
# define UPLINK_SESSION_SETTING(Type, Name, name) \
    case SessionSetupActionKind_Set##Name: \
        return new Set##Name##SessionSetupAction();
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING
    
    default:
        return 0;
    }
}

inline const SessionSetup&
SessionSetup::
defaults ()
{
    static SessionSetup cfg;

    static bool initialized = false;

    if(initialized)
        return cfg;

    initialized = true;

    static const SessionSettings defaultSettings;

# define UPLINK_SESSION_SETTING(Type, Name, name) \
    { \
        cfg.actions.push_back(AnySessionSetupAction()); \
        cfg.actions.back().resetAs##Set##Name(defaultSettings.name); \
    }
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING

    return cfg;
}

//------------------------------------------------------------------------------

}
