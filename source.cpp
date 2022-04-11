#include "header.h"

uint64_t get_ms(void){
    struct timeval time_now{};
    gettimeofday(&time_now, nullptr);
    return (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000);;
}

void print_time_from_ms(uint64_t ms){
    char* str;
    uint32_t minutes = (ms / 1000) / 60;
    double seconds = (double)(ms % 60000) / 1000;
    printf("\nTraining time: %d minutes, %.2f seconds\n", minutes, seconds);
    return;
}
