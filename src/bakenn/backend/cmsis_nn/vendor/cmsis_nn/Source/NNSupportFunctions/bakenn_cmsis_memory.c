/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Freestanding memory primitives for the pinned BakeNN CMSIS-NN bundle.
 * Volatile byte accesses prevent a freestanding compiler from recognizing
 * these loops and lowering them back to undefined memcpy/memset symbols.
 */

#include <stddef.h>

#if defined(BAKENN_CMSIS_NN_FREESTANDING)

void *bakenn_cmsis_memcpy(void *destination, const void *source, size_t size)
{
    volatile unsigned char *target = (volatile unsigned char *)destination;
    const volatile unsigned char *input = (const volatile unsigned char *)source;
    for (size_t index = 0; index < size; ++index)
    {
        target[index] = input[index];
    }
    return destination;
}

void *bakenn_cmsis_memset(void *destination, int value, size_t size)
{
    volatile unsigned char *target = (volatile unsigned char *)destination;
    const unsigned char byte = (unsigned char)value;
    for (size_t index = 0; index < size; ++index)
    {
        target[index] = byte;
    }
    return destination;
}

#endif
