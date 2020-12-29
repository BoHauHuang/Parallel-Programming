TARGET := pi.out
CPP := pi.c

#CXX := clang++-11
#CXX := g++
CXX := gcc
#CFLAGS := -std=c++11 -Ofast -Wall
CFLAGS := -std=gnu99 -Ofast
CFLAGS += -ffast-math -mavx -mavx2 -mfma -finline-functions
#CFLAGS += -lstdc++ -static-libstdc++
#CFLAGS += -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
THREAD := -pthread

all: $(CPP)
	$(CXX) $(CPP) -o $(TARGET) $(CFLAGS) $(THREAD)

clean:
	rm *.out *.o
