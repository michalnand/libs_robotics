LIBS_PATH = $(HOME)/programming/libs_robotics
GCCPREFIX=
OUTPUT	 =bin/build


INC = -I .
INC+= -I $(LIBS_PATH)/LibsEmbedded
INC+= -I $(LIBS_PATH)/LibsDevices

CXX_SRCS+= $(shell find $(LIBS_PATH)/LibsEmbedded -name '*.cpp')
CXX_SRCS+= $(shell find . -name '*.cpp')


CXX = $(GCCPREFIX)g++
LD = $(GCCPREFIX)g++
OBJCOPY = $(GCCPREFIX)objcopy
OBJDUMP = $(GCCPREFIX)objdump
SIZE = $(GCCPREFIX)size

CXXFLAGS = -c -O3 -Wall -Wextra -pedantic -std=c++11 $(INC)

LDFLAGS = -O3



CXX_OBJS = $(CXX_SRCS:.cpp=.o)

all: $(OUTPUT)
	$(OBJDUMP) --disassemble $<.elf > $(OUTPUT).asm
	$(SIZE) $(OUTPUT).elf

$(OUTPUT): $(CXX_OBJS)
	$(LD) $(LDFLAGS) -o $@.elf $(CXX_OBJS) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

CLEAN_FILES = $(CXX_OBJS)
CLEAN_FILES+= $(OUTPUT)*

clean:
	rm $(CLEAN_FILES) -f
