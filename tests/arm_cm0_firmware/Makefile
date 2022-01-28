STD_PERIPH_PATH = $(HOME)/programming/libs_robotics/LibsARM/stm32f0/

GCCPREFIX	=arm-none-eabi-
OUTPUT	 	=bin/build

#TARGET 		= -mcpu=cortex-m0 -mthumb -DSTM32F051 -DUSE_STDPERIPH_DRIVER
TARGET 		= -mcpu=cortex-m0 -mthumb -DSTM32F042 -DUSE_STDPERIPH_DRIVER
LDSCRIPT 	= ./device/ldscript.ld

INC+= -I $(STD_PERIPH_PATH)/CMSIS/Include/
INC+= -I $(STD_PERIPH_PATH)/CMSIS/Device/ST/STM32F0xx/Include
INC+= -I $(STD_PERIPH_PATH)/STM32F0xx_StdPeriph_Driver/inc

INC+= -I ./device
INC+= -I .


CXX_SRCS = $(shell find . -name '*.cpp')
C_SRCS = $(shell find  ./device -name '*.c')
C_SRCS+= $(shell find  $(STD_PERIPH_PATH)/STM32F0xx_StdPeriph_Driver/src -name '*.c')


CC 		= $(GCCPREFIX)gcc
CXX 	= $(GCCPREFIX)g++
LD 		= $(GCCPREFIX)g++
OBJCOPY = $(GCCPREFIX)objcopy
OBJDUMP = $(GCCPREFIX)objdump
SIZE 	= $(GCCPREFIX)size


LDFLAGS = $(TARGET) -Wl,--gc-sections

CCFLAGS = $(TARGET) $(INC) -c -O3 -Os -s -Wall -Wextra -pedantic -std=c99 -nostdlib   -fdata-sections -ffunction-sections   -Wl,--gc-sections
CXXFLAGS= $(TARGET) $(INC) -c -O3 -Os -s -Wall -Wextra -pedantic -std=c++14 -fno-rtti  -fno-exceptions -fdata-sections -ffunction-sections  -Wl,--gc-sections
LDFLAGS = -T$(LDSCRIPT) $(TARGET) -nostdlib -Wl,--gc-sections -lgcc


C_OBJS 		= $(C_SRCS:.c=.o)
CXX_OBJS 	= $(CXX_SRCS:.cpp=.o)


all: $(OUTPUT)
	$(OBJDUMP) --disassemble $<.elf > $(OUTPUT).asm
	$(OBJCOPY) -O ihex $<.elf $(OUTPUT).hex
	$(OBJCOPY) -O binary $<.elf $(OUTPUT).bin
	$(SIZE) $(OUTPUT).elf

$(OUTPUT): $(C_OBJS) $(CXX_OBJS) 
	$(LD) -o $@.elf $(C_OBJS) $(CXX_OBJS) $(LDFLAGS)


%.o: %.c
	$(CC) $(CCFLAGS) $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@




CLEAN_FILES = $(C_OBJS)
CLEAN_FILES+= $(CXX_OBJS)
CLEAN_FILES+= $(OUTPUT)*

clean:
	rm $(CLEAN_FILES) -f
