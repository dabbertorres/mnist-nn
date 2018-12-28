# compiler config
CXX       := g++
CXX_FLAGS := -Wall -Wextra -std=c++17
LD_FLAGS  := -lstdc++fs

# directory structure
SRC_DIR   := neural_network
BUILD_DIR := build
MNIST_DIR := mnist

# source and build output
OUT := $(BUILD_DIR)/neural_network
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
DEP := $(OBJ:%.o=%.d)

# mnist data
MNIST_URL     := http://yann.lecun.com/exdb
TRAIN_SET_LBL := $(MNIST_DIR)/train-labels-idx1-ubyte
TRAIN_SET_IMG := $(MNIST_DIR)/train-images-idx3-ubyte
TEST_SET_LBL  := $(MNIST_DIR)/t10k-labels-idx1-ubyte
TEST_SET_IMG  := $(MNIST_DIR)/t10k-images-idx3-ubyte
MNIST_FILES   := $(TRAIN_SET_LBL) $(TRAIN_SET_IMG) $(TEST_SET_LBL) $(TEST_SET_IMG)

# misc tools
GZIP := gzip
CURL := curl

.PHONY: clean release debug mnist-dl

release: OPT_FLAGS ?= -O2
debug:   OPT_FLAGS ?= -O0 -g

$(OUT): $(OBJ)
	$(CXX) $(CXX_FLAGS) $(OPT_FLAGS) $^ -o $@ $(LD_FLAGS)

$(OBJ): $(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXX_FLAGS) $(OPT_FLAGS) -c -MMD $^ -o $@

-include $(DEP)

$(BUILD_DIR):
	mkdir -p $@

mnist-dl: $(MNIST_FILES)

$(MNIST_FILES): %: %.gz
	$(GZIP) -d $<

$(addsuffix .gz, $(MNIST_FILES)): | $(MNIST_DIR)
	$(CURL) -o $@ $(MNIST_URL)/$@

$(MNIST_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
