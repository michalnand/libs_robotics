#ifndef _MATH_H_
#define _MATH_H_

#include "math_t.h"

template<class DType>
struct Vect2d
{   
    DType x, y;
};

template<class DType>
struct Vect3d
{   
    DType x, y, z;
};

template<class DType>
struct Vect4d
{   
    DType x, y, z, w;
};


float sin(x);
float cos(x);
float tan(x);

float asin(x);
float acos(x);
float atan(x);

float atan2(float x, float y);

#endif