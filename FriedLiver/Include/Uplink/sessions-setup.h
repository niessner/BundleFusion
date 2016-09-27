//
//  network/sessions-setup.h
//  Uplink
//
//  Copyright (c) 2015 Occipital, Inc. All rights reserved.
//

# pragma once

# include "./sessions-settings.h"
# include "./core/serializers.h"
# include "./core/enum.h"
# include <vector>

namespace uplink {

//------------------------------------------------------------------------------

struct SessionSetupAction : Serializable, Clonable<SessionSetupAction>
{
    virtual bool applyTo (SessionSettings& sessionSettings) const = 0;
};

typedef uplink_ref<SessionSetupAction> SessionSetupActionRef;

//------------------------------------------------------------------------------

UPLINK_ENUM_BEGIN(SessionSetupActionKind)
# define UPLINK_SESSION_SETTING(Type, Name, name) \
    SessionSetupActionKind_Set##Name,
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING
UPLINK_ENUM_END(SessionSetupActionKind)

static SessionSetupAction* newSessionSetupAction (SessionSetupActionKind kind);

//------------------------------------------------------------------------------

# define UPLINK_SESSION_SETTING(Type, Name, name) \
struct Set##Name##SessionSetupAction : SessionSetupAction \
{ \
    Set##Name##SessionSetupAction () {} \
    Set##Name##SessionSetupAction (const Type& value) : value(value) {} \
    Set##Name##SessionSetupAction (const Set##Name##SessionSetupAction& copy) : value(copy.value) {} \
    virtual SessionSetupAction* clone () const { return new Set##Name##SessionSetupAction(*this); } \
    virtual bool applyTo (SessionSettings& sessionSettings) const { sessionSettings.set##Name(value); return true; } \
    virtual bool serializeWith (Serializer& serializer) { return serializer.put(value); } \
    Type value; \
};
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING

//------------------------------------------------------------------------------

struct AnySessionSetupAction : SessionSetupAction
{
    AnySessionSetupAction ();
    AnySessionSetupAction (const AnySessionSetupAction& copy);

    void swapWith (AnySessionSetupAction& other);

    void clear ();

# define UPLINK_SESSION_SETTING(Type, Name, name) \
    void resetAsSet##Name (const Type& value) \
    { \
        kind = SessionSetupActionKind_Set##Name; \
        that.reset(new Set##Name##SessionSetupAction(value)); \
    }
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING

    virtual SessionSetupAction* clone () const { return new AnySessionSetupAction(*this); }

    virtual bool serializeWith (Serializer& serializer);

    virtual bool applyTo (SessionSettings& sessionSettings) const;

    SessionSetupActionKind kind;
    SessionSetupActionRef  that;
};

//------------------------------------------------------------------------------

struct SessionSetup : SessionSetupAction, Message
{
    UPLINK_MESSAGE_CLASS(SessionSetup)

    static const SessionSetup& defaults ();

    SessionSetup ();

    void swapWith (SessionSetup& other);

    virtual bool serializeWith (Serializer &serializer);

    virtual bool applyTo (SessionSettings& sessionSettings) const;
    
# define UPLINK_SESSION_SETTING(Type, Name, name) \
    void addSet##Name##Action (const Type& value) \
    { \
        actions.push_back(AnySessionSetupAction()); \
        actions.back().resetAsSet##Name(value); \
    }
         UPLINK_SESSION_SETTINGS()
# undef  UPLINK_SESSION_SETTING

    std::vector<AnySessionSetupAction> actions;
};

//------------------------------------------------------------------------------

}

# include "./sessions-setup.hpp"
