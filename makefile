# Makefile for mCUDA - Simple CUDA Library
# Author: HS4T

# Compilers
NVCC = nvcc
CXX  = g++

# File paths
INCLUDE_DIR = include
SRC_DIR = src
TEST_DIR = test

# Source files
CU_SRC = $(SRC_DIR)/mCUDA.cu
CPP_SRC = $(SRC_DIR)/mCUDA.cc

# Object files
CU_OBJ = mCUDA.o
CPP_OBJ = mCUDA_host.o
LIB = libmCUDA.a

# Test files
TEST_SRC = $(TEST_DIR)/main.cc
TEST_BIN = $(TEST_DIR)/main

# Build the static library
all: $(LIB)

$(CU_OBJ): $(CU_SRC)
	$(NVCC) -I$(INCLUDE_DIR) -c $< -o $@

$(CPP_OBJ): $(CPP_SRC)
	$(CXX) -I$(INCLUDE_DIR) -c $< -o $@

$(LIB): $(CU_OBJ) $(CPP_OBJ)
	ar rcs $(LIB) $(CU_OBJ) $(CPP_OBJ)

# Build test binary with g++
$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(CXX) $(TEST_SRC) -I$(INCLUDE_DIR) -L. -lmCUDA -L/usr/local/cuda/lib64 -lcudart -o $(TEST_BIN)

test: $(TEST_BIN)

run: $(TEST_BIN)
	./$(TEST_BIN)

# Clean all generated files
clean:
	rm -f $(CU_OBJ) $(CPP_OBJ) $(LIB) $(TEST_BIN)
