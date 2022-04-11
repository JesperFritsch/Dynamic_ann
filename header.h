#if !defined(HEADER_H_)
#define HEADER_H_

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <time.h>
#include <fstream>
#include <sys/time.h>
#include <ctime>
#include <math.h>

#include "Ann.h"

uint64_t get_ms(void);
void print_time_from_ms(uint64_t ms);



#endif // HEADER_H_