# Makefile for mCUDA - Simple CUDA Library
# Author: HS4T

# Compiler
NVCC = nvcc

# File paths
INCLUDE_DIR = include
SRC_DIR = src
TEST_DIR = test

# Files
CU_SRC = $(SRC_DIR)/mCUDA.cu
OBJ = mCUDA.o
LIB = libmCUDA.a
TEST_SRC = $(TEST_DIR)/main.cc
TEST_BIN = $(TEST_DIR)/main

# Build the static library
all: $(LIB)

$(OBJ): $(CU_SRC)
	$(NVCC) -I$(INCLUDE_DIR) -c $(CU_SRC) -o $(OBJ)

$(LIB): $(OBJ)
	ar rcs $(LIB) $(OBJ)

# Build and run test
test: all
	$(NVCC) -I$(INCLUDE_DIR) $(TEST_SRC) -L. -lmCUDA -o $(TEST_BIN)

run: test
	./$(TEST_BIN)

# Clean all generated files
clean:
	rm -f $(OBJ) $(LIB) $(TEST_BIN)
