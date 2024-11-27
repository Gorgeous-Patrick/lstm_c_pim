#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <alloc.h>
#include <assert.h>

#define KILL_PROCESS exit(EXIT_FAILURE)
#define DEBUG_MSG ;
#define ERROR_MGS(...) ;
#define RAND_MAX 32767

#define PANIC(...) ({ERROR_MGS(__VA_ARGS__); DEBUG_MSG; KILL_PROCESS;})

#define SAFE_MALLOC(size)({\
    void * memory = mem_alloc(size); \
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

#endif