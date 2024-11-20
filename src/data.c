#include "data.h"
#include "assert.h"

struct Data{
    double * ptr;
    int size;
    struct ref refcount;
};

static inline double * default_alloc(size_t size){
    return (double *)SAFE_MALLOC(size);
}

void data_dec(Data * self){
    ref_dec(&self->refcount);
}

void data_inc(Data * self){
    ref_inc(&self->refcount);
}

void data_free(const struct ref *ref){
    Data * data = container_of(ref, Data, refcount);
    if(data != NULL){
        SAFE_FREE(data->ptr);
        SAFE_FREE(data);
    }
    

    data = NULL;
}

Data * data_init(int size){
    return data_init_with_allocator(size, default_alloc);
}

Data * data_init_with_allocator(int size, allocator alloc){
    assert(size > 0);

    Data * data = (Data *)SAFE_MALLOC(sizeof(Data));
    data->size = size;
    data->refcount = (struct ref){.count = 1, .free = data_free};
    data->ptr = alloc(sizeof(double) * size);
    return data;   
}

void data_insert(Data * self, double value, int index){
    assert(index < self->size);

    if(self->ptr == NULL){
        self->ptr = (double *)SAFE_MALLOC(self->size);
    }

    self->ptr[index] = value;
}

void data_memcpy(Data * dest, Data * src, int dest_offset, int src_offset, int length){
    if(dest->ptr == NULL){
        dest->ptr = SAFE_MALLOC(sizeof(double) * dest->size);
    }

    for(int i = 0; i < length; i++){
        dest->ptr[i + dest_offset] = src->ptr[i + src_offset];
    }
}

double data_get(Data * self, int index){
    assert(index < self->size);
    assert(self->ptr != NULL);

    return self->ptr[index];
}

double * data_raw_ptr(Data * self){
    return self->ptr;
}

void data_assign_ptr(Data * self, double * ptr){
    self->ptr = ptr;
}
