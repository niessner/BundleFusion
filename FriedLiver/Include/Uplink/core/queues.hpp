//
//  system/queues.hpp
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./queues.h"

namespace uplink {

//------------------------------------------------------------------------------

template < typename Type >
struct UpdateBySwap
{
    static void perform (Type& first, Type& second)
    {
        uplink_swap(first, second);
    }
};

template < typename Type >
struct UpdateByCopy
{
    static void perform (const Type& source, Type& destination)
    {
        destination = source;
    }
};

//------------------------------------------------------------------------------

# define QUEUE_METHOD(...) \
template < class Item > \
inline __VA_ARGS__ \
Queue<Item>::
  
QUEUE_METHOD()
Queue () // Do not drop anything by default.
: count(0)
, items(0)
, maximumSize(0)
, droppingStrategy(RandomOneDroppingStrategy)
{
}

QUEUE_METHOD()
~Queue ()
{
}

QUEUE_METHOD(void)
pushByCopy (const Item& newItem)
{
    push< UpdateByCopy<Item> >(const_cast<Item&>(newItem));
}

QUEUE_METHOD(void)
pushBySwap (Item& newItem)
{
    push< UpdateBySwap<Item> >(newItem);
}

QUEUE_METHOD(bool)
popBySwap (Item& lastItem)
{
    Items tmp;
    
    {
        const MutexLocker _(mutex);
        
        if (items.empty())
            return false;
        
        tmp.splice(tmp.begin(), items, items.begin());
        --count;
    }
    
    uplink_swap(tmp.front(), lastItem);
    
    // FIXME: Recycle node.
    
    poppingRate.tick();
    
    return true;
}

QUEUE_METHOD(void)
isEmpty ()
{
    return 0 == count;
}

QUEUE_METHOD(float)
getUsageRatio () const
{
    const MutexLocker _(mutex);
    
    return 0 == count ? 0.f : float(count) / float(maximumSize);
}

QUEUE_METHOD(void)
setMaximumSize (int newMaximumSize)
{
    assert(0 <= newMaximumSize);
    
    // MaximumSize == 0 means: no dropping at all.
    if (0 < newMaximumSize && newMaximumSize < count)
    {
        const int excess = count - newMaximumSize;
        ItemsIterator i = items.begin();
        std::advance(i, excess);
        items.erase(items.begin(), i);
        count = newMaximumSize;
    }
    
    maximumSize = newMaximumSize;
}

QUEUE_METHOD(void)
setDroppingStategy (DroppingStrategy newStrategy)
{
    droppingStrategy = newStrategy;
}

QUEUE_METHOD(void)
reset ()
{
    const MutexLocker _(mutex);
    
    items.clear();
    count = 0;
    
    pushingRate.reset();
    poppingRate.reset();
}

QUEUE_METHOD(double)
currentPushingRate () const
{
    const MutexLocker _(mutex);
    
    return pushingRate.windowedRate();
}

QUEUE_METHOD(double)
currentPoppingRate () const
{
    const MutexLocker _(mutex);
    
    return poppingRate.windowedRate();
}

template < class Item >
template < class UpdateMethod >
inline void
Queue<Item>::
push (Item& newItem)
{
    Items tmp;
    
    // FIXME: Use recycled node. This code is intended to minimize the number of allocations
    // by preallocating a list node element. Right now there is no preallocation, so
    // push_back would be equivalent. However, the final splice is likely to be faster than
    // the full push_back, so it slightly reduces the time we spent with the mutex locked.
    
    ItemsIterator i = tmp.insert(tmp.end(), Item());

    UpdateMethod::perform(newItem, *i);
    
    {
        const MutexLocker _(mutex);
        
        // Zero-limit means: do not ever drop anything.
        if (0 < maximumSize && count == maximumSize)
            drop();
        
        items.splice(items.end(), tmp);
        
        ++count;
    }
    
    pushingRate.tick();
}
    
QUEUE_METHOD(void)
dropRandomly (int howMany)
{
    // Assuming the mutex is already locked.
    
    while (0 < howMany)
    {
        const int index = int(float(rand()) / float(RAND_MAX + 1) * float(count));
        ItemsIterator i = items.begin();
        std::advance(i, index);
        items.erase(i); // FIXME: Recycle node.
        --count;
        --howMany;
    }
}
    
QUEUE_METHOD(void)
drop ()
{
    // Assuming the mutex is already locked.
    
    switch (droppingStrategy)
    {
        case OldestOneDroppingStrategy:
        {
            items.pop_front(); // FIXME: Recycle node.
            --count;
        }
            break;
            
        case RandomOneDroppingStrategy:
        {
            dropRandomly();
        }
            break;
            
        case RandomFivePercentDroppingStrategy:
        {
            const int howMany = std::max(1, maximumSize / 20);
            dropRandomly(howMany);
        }
            
        case OneEveryTenDroppingStrategy:
        {
            int newCount = count;
            ItemsIterator i = items.begin();
            for (int n = 0; n < count; ++n)
            {
                if (0 != n % 10)
                {
                    ++i;
                    continue;
                }
                
                items.erase(i++); // FIXME: Recycle node.
                --newCount;
            }
            count = newCount;
        }
            break;
    }
}
    
//------------------------------------------------------------------------------

}
