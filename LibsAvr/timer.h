#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdint.h>

class Timer
{
  public:
    Timer();

    uint64_t get();
};

#endif
