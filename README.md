## Andes Neural Network Library
___
- Andes Neural Network library user manual  
    The user manual of Andes Neural Network library can be found under docs/html.
- For non-Andes toolchains
    - Steps to build the Andes Neural Network library  
        Run ***build.sh*** with the compiler name as the argument. For example:  
        `./build.sh "riscv32-unknown-elf-gcc"`  
        Once the build is complete, a static library named ***libnn.a*** will be generated. For more usage details, please refer to the comments at the beginning of ***build.sh***.
    - Steps to build the example
        1. Navigate to *example* folder
        2. Build the example with the following command (assumes *riscv32-unknown-elf-gcc* as the compiler):  
        `riscv32-unknown-elf-gcc example.c -o example.out -lnn -L../ -I../Include`
- For Andes toolchains
    - Steps to build and install the Andes Neural Network library  
        Execute the ***build_default_library.sh*** script. For example:  
        `./build_default_library.sh <RISCV32-UNKNOWN-ELF_TOOLCHAIN_PATH>/riscv32-elf/lib riscv32-elf <RISCV32-UNKNOWN-ELF_TOOLCHAIN_PATH>/riscv32-elf/include gcc`
    - Steps to build the example
        1. Navigate to *example* folder
        2. Build the example with the following command (assumes *riscv32-unknown-elf-gcc* as the compiler):  
        `riscv32-unknown-elf-gcc example.c -o example.out -lnn`