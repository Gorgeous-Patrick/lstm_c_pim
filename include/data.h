#ifndef DATA_H
#define DATA_H

#include "utils.h"

typedef struct Data Data;

typedef double * (*allocator)(size_t);

void data_dec(Data * self);
void data_inc(Data * self);
void data_free(const struct ref *ref);

/**
 * Create new data object with allocated memory
 */
Data * data_init(int size);
Data * data_init_with_allocator(int size, allocator alloc);
void data_insert(Data * self, double value, int index);
void data_memcpy(Data * dest, Data * src, int dest_offset, int src_offset, int length);
double data_get(Data * self, int index);
double * data_raw_ptr(Data * self);
void data_assign_ptr(Data * self, double * ptr);
#endif // DATA_H