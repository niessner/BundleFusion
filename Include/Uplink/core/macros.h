//
//  core/macros.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

// Ubiquitous, code-shortening platform-independent preprocessor macros.

# pragma once

//------------------------------------------------------------------------------

# define header_only void __dummy_symbol__##__COUNTER__ () {}

# define zero_delete(Ptr) delete (Ptr); (Ptr) = 0;

# define sizeof_array(Array) sizeof(Array) / sizeof(Array[0])

# define non_copyable(Class)              \
private:                                  \
    Class (const Class& copy);            \
    Class& operator = (const Class& rhs);

# define unimplemented assert(false)

// FIXME: Generate all the combinations, eventually.

# define break_if(...)     if ( (__VA_ARGS__)) break;
# define break_unless(...) if (!(__VA_ARGS__)) break;

# define continue_if(...)      if ( (__VA_ARGS__)) continue;
# define continue_unless(...)  if (!(__VA_ARGS__)) continue;

# define return_if(...)           if ( (__VA_ARGS__)) { return; }
# define return_unless(...)       if (!(__VA_ARGS__)) { return; }

# define return_true_if(...)      if ( (__VA_ARGS__)) { return true; }
# define return_true_unless(...)  if (!(__VA_ARGS__)) { return true; }

# define return_false_if(...)     if ( (__VA_ARGS__)) { return false; }
# define return_false_unless(...) if (!(__VA_ARGS__)) { return false; }

# define return_zero_if(...)      if ( (__VA_ARGS__)) { return 0; }
# define return_zero_unless(...)  if (!(__VA_ARGS__)) { return 0; }

# define return_if_zero(...)           if (0 == (__VA_ARGS__)) { return; }
# define return_if_non_zero(...)       if (0 != (__VA_ARGS__)) { return; }

# define return_true_if_zero(...)      if (0 == (__VA_ARGS__)) { return true; }
# define return_true_if_non_zero(...)  if (0 != (__VA_ARGS__)) { return true; }

# define return_false_if_zero(...)     if (0 == (__VA_ARGS__)) { return false; }
# define return_false_if_non_zero(...) if (0 != (__VA_ARGS__)) { return false; }

# define return_zero_if_zero(...)      if (0 == (__VA_ARGS__)) { return 0; }
# define return_zero_if_non_zero(...)  if (0 != (__VA_ARGS__)) { return 0; }

# define report_continue_if(What, ...)      if ( (__VA_ARGS__)) { uplink_log_error(What); continue; }
# define report_continue_unless(What, ...)  if (!(__VA_ARGS__)) { uplink_log_error(What); continue; }

# define report_if(What, ...)       if ((__VA_ARGS__)) { uplink_log_error(What); return; }
# define report_false_if(What, ...) if ((__VA_ARGS__)) { uplink_log_error(What); return false; }
# define report_zero_if(What, ...)  if ((__VA_ARGS__)) { uplink_log_error(What); return 0; }

# define report_unless(What, ...)       if (!(__VA_ARGS__)) { uplink_log_error(What); return; }
# define report_false_unless(What, ...) if (!(__VA_ARGS__)) { uplink_log_error(What); return false; }
# define report_zero_unless(What, ...)  if (!(__VA_ARGS__)) { uplink_log_error(What); return 0; }

# define report_if_zero(What, ...)       if (0 == (__VA_ARGS__)) { uplink_log_error(What); return; }
# define report_if_non_zero(What, ...)   if (0 != (__VA_ARGS__)) { uplink_log_error(What); return; }

# define report_false_if_zero(What, ...)     if (0 == (__VA_ARGS__)) { uplink_log_error(What); return false; }
# define report_false_if_non_zero(What, ...) if (0 != (__VA_ARGS__)) { uplink_log_error(What); return false; }

# define report_zero_if_zero(What, ...)      if (0 == (__VA_ARGS__)) { uplink_log_error(What); return 0; }
# define report_zero_if_non_zero(What, ...)  if (0 != (__VA_ARGS__)) { uplink_log_error(What); return 0; }

//------------------------------------------------------------------------------
