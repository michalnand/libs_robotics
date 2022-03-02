#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdint.h>

#define TIMER_MAX_TASKS     ((uint32_t)32)


class Task
{
  public:
    Task() { };
    virtual ~Task() { }; 
    virtual void main() { };
}; 

class Timer
{
    public:
        Timer();
        virtual ~Timer();

        void init(uint32_t frequency = 1000);

        //@brief add periodic task
        //@param callback task function callback, void my_func(void)
        //@param period_ms executing period in miliseconds
        //if set to false, task function is executing inside interrupt rutine - and other interrupts are blocked
        int add_task(class Task *callback_class, unsigned int period_ms);

    public:
        uint32_t get_time();
        void delay_ms(uint32_t time_ms);

    private:
        uint32_t current_ptr;
};

#endif