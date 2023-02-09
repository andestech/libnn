#include <stdio.h>
#include "riscv_nn_activation.h"
#include "data.h"

#define SIZE 32

int main()
{
    int i;
    q7_t *in_out = data;

    printf("data before relu:\n");
    for (i=0; i<SIZE; i++)
        printf("%02x ", (unsigned char)in_out[i]);

    riscv_nn_relu_s8(in_out, SIZE);

    printf("\n\ndata after relu:\n");
    for (i=0; i<SIZE; i++)
        printf("%02x ", (unsigned char)in_out[i]);
    printf("\n");

    return 0;
}
