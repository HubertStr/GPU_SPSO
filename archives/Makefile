TARGET   = program

C_SRCS   = support.c
CXX_SRCS = main_CPU_prototype.c

OBJS     = $(C_SRCS:.c=.o)
OBJS    += $(CXX_SRCS:.c=.o)

CFLAGS   = -g -Wall 
CXXFLAGS = -g -Wall
LDFLAGS  = -lpcap

$(TARGET) : $(OBJS)
	(CXX) -o $@ $^ $(LDFLAGS)

clean :
	(RM) $(OBJS) $(TARGET)
