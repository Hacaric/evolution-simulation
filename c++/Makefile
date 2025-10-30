# The C++ compiler to use
CXX = g++

# Compiler flags:
# -std=c++20: Use the C++20 standard
# -Wall:       Enable all compiler warnings (recommended)
# -Wextra:     Enable extra compiler warnings (recommended)
# -O2:         Enable optimizations for performance
CXXFLAGS = -std=c++20 -Wall -Wextra -O1

# The name of the target executable
TARGET = main

# The source file(s) for the project
SRCS = main.cpp neurons.cpp dataset.cpp

# The default target, 'all', builds the main target.
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

.PHONY: all clean

# Target to remove the built executable
clean:
	rm -f $(TARGET)
