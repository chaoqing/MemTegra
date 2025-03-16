# MemTegra

MemTegra is a modern C++ library designed to provide unified memory management for across CPU and GPU.

## Features

- **CUDA Support**: Designed to work seamlessly with CUDA, making it suitable for GPU programming.
- **Utility Functions**: Includes various utility functions to assist with memory management and other tasks.

## Installation

To build and install MemTegra, you need to have CMake and a compatible C++ compiler installed. Follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/chaoqing/MemTegra.git
   cd MemTegra
   ```

2. Create a build directory:
   ```
   mkdir build
   cd build
   ```

3. Run CMake to configure the project:
   ```
   cmake ..
   ```

4. Build the project:
   ```
   make
   ```

5. Optionally, run tests:
   ```
   make test
   ```

## Usage

To use the MemTegra library in your project, include the header file:

```cpp
#include "MemTegra/MemTegra.h"
```

You can then use the `mallocAligned` and `freeAligned` methods to manage memory:

```cpp
MemTegra memTegra;
void* ptr = memTegra.mallocAligned(size);
memTegra.freeAligned(ptr);
```

## Testing

MemTegra includes unit tests to ensure the correctness of memory allocation and deallocation. The tests can be run using the following command:

```
make test
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.