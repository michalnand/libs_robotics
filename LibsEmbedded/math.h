#ifndef _MATH_H_
#define _MATH_H_

#include "math_t.h"

#define PI ((float)3.141592654)


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


float sin(float x);
float cos(float x);
float tan(float x);

float asin(float x);
float acos(float x);
float atan(float x);

float atan2(float x, float y);

#endif