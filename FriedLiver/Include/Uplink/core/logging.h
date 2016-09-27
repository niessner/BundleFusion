# pragma once

# include "./types.h"

namespace uplink {

//------------------------------------------------------------------------------

enum Verbosity {

    InvalidVerbosity = -1,

    Debug,
    Info,
    Warning,
    Error,

    NumVerbosities
};

# define UPLINK_LOG_PREFIX "UPLINK: "

// Thread-safe catch-all.
void log_line (Verbosity verbosity, CString message);

inline CString log_prefix (Verbosity verbosity)
{
    assert(int(InvalidVerbosity) < int(verbosity) && int(verbosity) < int(NumVerbosities));

    static const CString prefixes [NumVerbosities] =
    {
        UPLINK_LOG_PREFIX "DEBUG   : ",
        UPLINK_LOG_PREFIX "INFO    : ",
        UPLINK_LOG_PREFIX "WARNING : ",
        UPLINK_LOG_PREFIX "ERROR   : ",
    };

    return prefixes[verbosity];
}

inline void log_debug   (CString message) { log_line(Debug  , message); }
inline void log_info    (CString message) { log_line(Info   , message); }
inline void log_warning (CString message) { log_line(Warning, message); }
inline void log_error   (CString message) { log_line(Error  , message); }

# if UPLINK_DEBUG
#   define uplink_log_debug(...)   ::uplink::log_debug(  ::uplink::formatted(__VA_ARGS__).get())
#else
#   define uplink_log_debug(...)
# endif
# define   uplink_log_info(...)    ::uplink::log_info(   ::uplink::formatted(__VA_ARGS__).get())
# define   uplink_log_warning(...) ::uplink::log_warning(::uplink::formatted(__VA_ARGS__).get())
# define   uplink_log_error(...)   ::uplink::log_error(  ::uplink::formatted(__VA_ARGS__).get())

//------------------------------------------------------------------------------

} // uplink namespace
