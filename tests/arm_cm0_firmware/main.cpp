#include "stm32f0xx.h"
#include "system_stm32f0xx.h"

volatile uint32_t g_time = 0;

#ifdef __cplusplus
extern "C" {
#endif

void SysTick_Handler(void)
{
  g_time++;
}

#ifdef __cplusplus
}
#endif
 


void delay_ms(uint32_t time_wait)
{
  time_wait = time_wait + g_time;

  while (time_wait > g_time)
  {
    __asm("wfi");
  }
} 


void SetPLL()
{
  RCC_PLLConfig(RCC_PLLSource_HSI, RCC_PLLMul_6);
  RCC_PLLCmd(ENABLE);

  // Wait for PLLRDY after enabling PLL.
  while (RCC_GetFlagStatus(RCC_FLAG_PLLRDY) != SET)
  { 
    __asm("nop");
  }

  RCC_SYSCLKConfig(RCC_SYSCLKSource_PLLCLK);  // Select the PLL as clock source.
  SystemCoreClockUpdate();

  SystemCoreClockUpdate();
}

 
int main(void)
{
  //cpu clock init
  SystemInit();

  // setup PLL, 6*HSI = 48MHz
  SetPLL(); 

  //interrupt every 48MHz/1000 cpu tics, 1ms
  SysTick_Config(SystemCoreClock/1000);
  __enable_irq();

  GPIO_InitTypeDef   GPIO_InitStructure;

  //enable GPIOB clock
  RCC_AHBPeriphClockCmd(RCC_AHBPeriph_GPIOB, ENABLE);

  //configure pin PB3 pin as output, push-pull
  GPIO_InitStructure.GPIO_Pin   = GPIO_Pin_3;
  GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_OUT;
  GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
  GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
  GPIO_InitStructure.GPIO_PuPd  = GPIO_PuPd_NOPULL;
  GPIO_Init(GPIOB, &GPIO_InitStructure);

  while(1) 
  {
    //led on
    GPIOB->ODR|= (1<<3);
    delay_ms(100);

    //led off
    GPIOB->ODR&= ~(1<<3);
    delay_ms(900);
  }

  return 0;
} 