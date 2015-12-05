//
//  core/memory.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include <functional>
# include <memory>

namespace uplinksimple {

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





//
//  core/memory.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./memory.h"

namespace uplinksimple {

//------------------------------------------------------------------------------

inline
MemoryBlock::MemoryBlock() :
_freeWhenDone(false), Size(0), AllocatedSize(0), Data(nullptr), _releaseFunction(nullptr)
{
}

inline
MemoryBlock::~MemoryBlock()
{
    this->freeMemory();
}

inline void
MemoryBlock::freeMemory()
{
    if(_freeWhenDone)
        deallocate((uint8_t*)this->Data); // Free memory, if requested to.

    callReleaseFunction();
}

inline void
MemoryBlock::swapWith (MemoryBlock& other)
{
    std::swap (Data, other.Data);
    std::swap (Size, other.Size);
    std::swap (AllocatedSize, other.AllocatedSize);
    std::swap (_freeWhenDone, other._freeWhenDone);
    std::swap (_releaseFunction, other._releaseFunction);
}

inline void
MemoryBlock::copyTo (MemoryBlock& other) const
{
    other.Resize(Size);

    if (Size > 0)
        memcpy (other.Data, Data, Size);
}

inline void
MemoryBlock::clear ()
{
    freeMemory();

    Data = nullptr;
    Size = 0;
    AllocatedSize = 0;
    _freeWhenDone = false;
    _releaseFunction = nullptr;
}

inline void
MemoryBlock::Resize(size_t newSize, bool preserveData, bool reclaimExtraIfShrinking)
{
    // Size is right already:
    if(Size == newSize) return;

    // Need to shrink, but don't need to reclaim, so set size & return:
    if(AllocatedSize >= newSize && !reclaimExtraIfShrinking)
    {
        Size = newSize;
        return;
    }

    // Below here, we need to actually allocate/reallocate memory:

    // Empty?  We can jump directly to malloc.
    if(Data == NULL && AllocatedSize == 0)
    {
        freeMemory();

        Data = allocate(newSize);
        AllocatedSize = newSize;
        Size = newSize;
        _freeWhenDone = true;
    }

    bool canFreeMemory = _freeWhenDone || _releaseFunction;

    if (!canFreeMemory)
        assert(false); // MemoryBlock needs to resize, but cannot free its current memory. Very likely a leak!

    if (_freeWhenDone)
    {
        // Use realloc if we're asked to preserve.  Otherwise, free+malloc.
        if(preserveData)
        {
            Data = reallocate(Data, newSize);
            Size = newSize;
            AllocatedSize = newSize;
        }
        else
        {
            deallocate(Data);

            // Allocate a new buffer.
            // Note that _freeWhenDone must remain true to manage this new memory.
            Data = allocate(newSize);
            Size = newSize;
            AllocatedSize = newSize;
        }
    }

    if (_releaseFunction)
    {
        assert(_freeWhenDone == false); // MemoryBlock cannot have both freeWhenDone and a release block.

        uint8_t * previousData = Data;
        size_t previousSize = Size;

        // If we don't need to preserve contents, release earlier (less max memory usage)
        if(!preserveData)
            callReleaseFunction();

        // Allocate a new buffer.
        Data = allocate(newSize);
        Size = newSize;
        AllocatedSize = newSize;
        _freeWhenDone = true; // We now must manage this buffer.

        if(preserveData) {
            memcpy(Data, previousData, previousSize);
            callReleaseFunction();
        }

    }
}

inline void
MemoryBlock::relinquishOwnership()
{
    assert(!_releaseFunction); // Extract with a release function is not supported yet.
    _freeWhenDone = false; // Caller will now be responsible for this memory
}

inline void
MemoryBlock::transferOwnership(ReleaseFunction& releaseFunction)
{
    if (_releaseFunction)
    {
        releaseFunction = _releaseFunction; // transfer the release function.
    }
    else
    {
        uint8_t* data = Data;
        releaseFunction = [data]() { deallocate(data); };
    }

    _releaseFunction = ReleaseFunction();

    _freeWhenDone = false; // Caller will now be responsible for this memory
}

//------------------------------------------------------------------------------

}

