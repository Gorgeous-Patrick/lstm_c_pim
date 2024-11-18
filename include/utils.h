#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - offsetof(type, member)))

#define KILL_PROCESS exit(EXIT_FAILURE)
#define DEBUG_MSG fprintf(stderr, "\n----------------\nDebug Info:\n Error in file %s at line %d\n", __FILE__, __LINE__);
#define ERROR_MGS(...) fprintf(stderr, __VA_ARGS__);

#define PANIC(...) ({ERROR_MGS(__VA_ARGS__); DEBUG_MSG; KILL_PROCESS;})

#define SAFE_MALLOC(size)({\
    void * memory = malloc(size); \
    if(memory == NULL){\
        PANIC("Memory could not be allocated");\
    }\
    memory;\
})

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof((x)[0]))

#define EXP(x) (((x + 3) * (x + 3)) + 3) / (((x-3) * (x-3)) + 3)

static inline double exp(double x){
    if(x < 0){
        return 1/(EXP(x));
    }else{
        return EXP(x);
    }
}

static inline double tanh(double x){
    return (EXP(x) + EXP(-x))/(EXP(x) - EXP(-x));
}

static inline double sigmoid(double x){
    return 1/(1 + EXP(-x));
}

/*
Reference counter via https://nullprogram.com/blog/2015/02/17/
*/
struct ref {
    void (*free)(const struct ref *);
    int count;
};

static inline void ref_inc(const struct ref *ref)
{
    ((struct ref *)ref)->count++;
}

static inline void ref_dec(const struct ref *ref)
{
    if (--((struct ref *)ref)->count == 0)
        ref->free(ref);
}

#endif