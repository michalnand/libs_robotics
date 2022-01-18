#include "timer.h"
#include <avr/io.h>
#include <avr/interrupt.h>


volatile uint64_t g_time;

#define MCU_FREQ  ((uint32_t)16000000)

Timer::Timer()
{
  // Set the Timer Mode to CTC
  TCCR0A |= (1 << WGM01);

  // Set the value that you want to count to
  OCR0A = ((uint32_t)MCU_FREQ)/(64*1000) - 1;
  //249;

  // set prescaler to 64 and start the timer
  TCCR0B = (1 << CS01)|(1 << CS00);

  TIMSK0 |= (1 << OCIE0A);    //Set the ISR COMPA vect

  g_time = 0;
  sei();
}

uint64_t Timer::get()
{
  cli();
  volatile uint64_t time = g_time;
  sei();

  return time;
}

ISR(TIMER0_COMPA_vect)
{
  g_time++;
}

