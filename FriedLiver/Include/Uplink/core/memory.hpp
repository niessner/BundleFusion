//
//  core/memory.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./memory.h"

namespace uplink {

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
