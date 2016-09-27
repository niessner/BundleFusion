//
//  core/memory.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include <functional>
# include <memory>

namespace uplink {

//------------------------------------------------------------------------------

class MemoryBlock
{
public: // Offer external access to the current allocation and deallocation routines, so that when we change our minds about these later on, we won't need to update dependent code.
    static uint8_t*   allocate (size_t size) { return (uint8_t*) std::malloc(size); }
    static uint8_t* reallocate (uint8_t* data, size_t size) { return (uint8_t*) std::realloc(data, size); }
    static void     deallocate (uint8_t* data) { std::free(data); }

public:

    // Public direct access to the data
    uint8_t * Data;
    size_t Size;
    size_t AllocatedSize; // total allocated size for Data. May be bigger than Size.

private:

    bool _freeWhenDone;
    
    typedef std::function<void(void)> ReleaseFunction;
    
    ReleaseFunction _releaseFunction;

    inline void callReleaseFunction()
    {
        if (_releaseFunction) {
            _releaseFunction(); // Call the release function, if specified.
            _releaseFunction = nullptr;
        }
    }

    void freeMemory();

public:

    // Destructor
    ~MemoryBlock();

    // Default constructor
    MemoryBlock();

    // Construct with already-existing data, and optionally take ownership and free when done.
    // We use a template for the function type to be able to catch ObjectiveC blocks and disable
    // them. Otherwise calling with a C++ lambda would be ambiguous.
    template <class FunctionParameterType>
    MemoryBlock(uint8_t* data, size_t size, bool freeWhenDone, FunctionParameterType releaseFunction)
    : _freeWhenDone(freeWhenDone), Size(size), AllocatedSize(size), Data(data), _releaseFunction(releaseFunction)
    {}

    // Important!  Here we delete the default copy constructor.  This ensures
    // that we don't accidentally end up with dual-ownership over the same
    // underlying memory.
    MemoryBlock(const MemoryBlock& other) = delete;

    void clear ();

    void swapWith (MemoryBlock& other);
    void copyTo (MemoryBlock& other) const;

    /// Transfers ownership of data to the caller. (Data & Size fields remain populated, however.)
    void relinquishOwnership();

    /// Transfers ownership of data to the caller, similarly to relinquishOwnership, but honoring the release function.
    void transferOwnership (ReleaseFunction& releaseFunction);

    // Construct with already-existing data, and optionally take ownership and free when done.
    // We use a template for the function type to be able to catch ObjectiveC blocks and disable
    // them. Otherwise calling with a C++ lambda would be ambiguous.
    /*
     Example usage:

     CFRetain(someCoreFoundationObjectRef);

     memoryBlock.ReplaceWith(dataPtr, dataLength, false , [someCoreFoundationObjectRef](){
     CFRelease(someCoreFoundationObjectRef);
     });
     */
    template <class FunctionParameterType>
    void ReplaceWith(uint8_t* data, size_t size, bool freeWhenDone, const FunctionParameterType& releaseFunction)
    {
        freeMemory();

        this->Data = data;
        this->Size = size;
        this->AllocatedSize = size; // FIXME: this may not be true, it may be bigger, we could take an additional parameter.
        this->_freeWhenDone = freeWhenDone;
        _releaseFunction = releaseFunction;
    }

    // Resize the memory block as efficiently as possible.
    void Resize(size_t size, bool preserveData = false, bool reallocIfShrinking = false);

#if __OBJC__
    // For now, we don't allow Objective C blocks, since in that case we would need to call Block_copy.
    // Use a C++11 lambda instead. Thus the following overloads are disabled for safety.
    // FIXME: we could actually implement these if we wanted to.
    MemoryBlock(uint8_t* data, size_t size, bool freeWhenDone, void(^releaseFunction)(void)) = delete;
    void ReplaceWith(uint8_t* data, size_t size, bool freeWhenDone, void(^releaseFunction)(void)) = delete;
#endif

public: // Convenience methods.
    uint8_t* begin() { assert(0 != Data);              return Data; }
    uint8_t* end  () { assert(0 != Data && 0 < Size);  return Data + Size; }
};

//------------------------------------------------------------------------------

}

# include "./memory.hpp"
