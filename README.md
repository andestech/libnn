## Andes Neural Network Library
___
- Andes Neural Network library user manual  
    The user manual of Andes Neural Network library could be found under docs/html.
- Steps to build the static Andes Neural Network library  
    Execute the ***build.sh*** with the compiler name as the parameter. For instance, you can use ***riscv32-unknown-elf-gcc*** to build the library with below command  
    `./build.sh "riscv32-unknown-elf-gcc"`  
    As the building is done, the static library named ***libnn.a*** will be generated. You can read the descriptions in the beginning of ***build.sh*** for more usage details.
- Steps to build the example
    1. navigate to *example* folder
    2. build the example with following command (assume using *riscv32-unknown-elf-gcc* as compiler)  
    `riscv32-unknown-elf-gcc example.c -o example.out -lnn -L../ -I../Include`