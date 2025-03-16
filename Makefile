# Variables
BUILD_DIR = build
CMAKE = cmake
MAKE = make

# Default target
all: build

# Build the project
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) .. && $(MAKE)

# Run tests
test: build
	@cd $(BUILD_DIR) && $(MAKE) test

# Clean build files
clean:
	@rm -rf $(BUILD_DIR)

# Rebuild the project
rebuild: clean build
