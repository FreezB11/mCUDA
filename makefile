NVCC = nvcc
FLAGS = -O2
SRC_DIR := src
BUILD_DIR := build

CU_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CU_FILES))

# Print file lists for debugging
$(info CU_FILES = $(CU_FILES))
$(info OBJ_FILES = $(OBJ_FILES))

all: $(BUILD_DIR)/main

$(BUILD_DIR)/main: $(OBJ_FILES)
	@echo "Linking $@"
	$(NVCC) $(FLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	@echo "Compiling $< -> $@"
	$(NVCC) $(FLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
