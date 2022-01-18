LIBS_PATH = $(HOME)/programming/libs_robotics
GCCPREFIX=avr-
OUTPUT	 =bin/build


INC = -I .
INC+= -I $(LIBS_PATH)/LibsAvr
INC+= -I $(LIBS_PATH)/LibsEmbedded
INC+= -I $(LIBS_PATH)/LibsDevices


CXX_SRCS = $(shell find $(LIBS_PATH)/LibsAvr -name '*.cpp')
CXX_SRCS+= $(shell find $(LIBS_PATH)/LibsEmbedded -name '*.cpp')
CXX_SRCS+= $(shell find $(LIBS_PATH)/LibsDevices -name '*.cpp')
CXX_SRCS+= $(shell find . -name '*.cpp')


TARGET = -mmcu=atmega328




CXX = $(GCCPREFIX)g++
LD = $(GCCPREFIX)g++
OBJCOPY = $(GCCPREFIX)objcopy
OBJDUMP = $(GCCPREFIX)objdump
SIZE = $(GCCPREFIX)size

CXXFLAGS = -c -O2 -Os -Wall -Wextra -pedantic -std=c++11 -fno-rtti -fno-exceptions $(INC) -g $(TARGET) -fdata-sections -ffunction-sections

LDFLAGS = $(TARGET) -Wl,--gc-sections



CXX_OBJS = $(CXX_SRCS:.cpp=.o)

all: $(OUTPUT)
	$(OBJDUMP) --disassemble $<.elf > $(OUTPUT).asm
	$(OBJCOPY) -O ihex $<.elf $(OUTPUT).hex
	$(OBJCOPY) -O binary $<.elf $(OUTPUT).bin
	$(SIZE) $(OUTPUT).elf

$(OUTPUT): $(CXX_OBJS)
	$(LD) $(LDFLAGS) -o $@.elf $(CXX_OBJS) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

CLEAN_FILES = $(CXX_OBJS)
CLEAN_FILES+= $(OUTPUT)*

clean:
	rm $(CLEAN_FILES) -f

burn: $(OUTPUT)
	avrdude -c arduino -p atmega328p -P /dev/ttyUSB0 -U flash:w:$(OUTPUT).hex
