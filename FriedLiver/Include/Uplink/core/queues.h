//
//  system/queues.h
//  Uplink
//
//  Copyright (c) 2013 Occipital. All rights reserved.
//

# pragma once

# include "./threads.h"
# include "./clocks.h"
# include <list>
# include <algorithm>
# include <iterator>

namespace uplink {

template < class Item >
class Queue
{
public:
    enum DroppingStrategy
    {
        OldestOneDroppingStrategy,
        RandomOneDroppingStrategy,
        RandomFivePercentDroppingStrategy,
        OneEveryTenDroppingStrategy
    };

public:
    Queue ();
    ~Queue ();
    
public:
    void pushByCopy (const Item& newItem);
    void pushBySwap (Item& newItem);

public:
    bool popBySwap (Item& lastItem);

public:
    void isEmpty ();
    float getUsageRatio () const;

public:
    void setMaximumSize (int newMaximumSize);
    void setDroppingStategy (DroppingStrategy newStrategy);

public:
    void reset ();

public:
    double currentPushingRate () const;
    double currentPoppingRate () const;

private:
    template < class UpdateMethod >
    void push (Item& newItem);

private:
    void dropRandomly (int howMany = 1);
    void drop ();

private:
    typedef std::list<Item> Items;
    typedef typename Items::iterator ItemsIterator;

private:
    mutable Mutex    mutex;
    int              count; // C++98 allows for o(n) std::list size implementations.
    Items            items;

private:
    int              maximumSize;
    DroppingStrategy droppingStrategy;

private:
    RateEstimator   pushingRate;
    RateEstimator   poppingRate;
};

}

# include "./queues.hpp"
