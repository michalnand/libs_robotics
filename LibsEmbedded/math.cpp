#include "math.h"
#include <stdint.h>


float sqrt(float x)
{
    if (x <= 0)
    {
        return 0;
    }

    float xhi = x;
    float xlo = 0;
    float guess = x/2;

    uint8_t steps = 0;

    while ( (abs(guess*guess - x) > 0.000000001) && (steps < 32) )
    {
        if (guess * guess > x)
        {
            xhi = guess;
        }
        else
        {
            xlo = guess;
        }

        float new_guess = (xhi + xlo) / 2;
        guess = new_guess;

        steps++;
    }
    
    return guess;
}


float sin(float x)
{
    int32_t cnt = x/PI;
    x = x - PI*cnt;

    return 0;
}

float cos(float x)
{

}

float tan(float x)
{
    return 0;
}

float asin(float x)
{
    x = clamp(x, -1.0f, 1.0f);
    return atan2(x, sqrt(1-x*x));
}

float acos(float x)
{
    x = clamp(x, -1.0f, 1.0f);
    return atan2(sqrt(1-x*x), x);
}

float atan(float x)
{ 
    return atan2(x, 1);
}

float atan2(float x, float y)
{
    if (x == 0 && y == 0)
    {
        return 0; 
    }

    auto a = min(abs(x), abs(y))/max(abs(x), abs(y));
    auto s = a*a;

    auto result = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;

    if (abs(y) > abs(x))
    {
        result = 1.57079637 - result;
    }

    if (x < 0)
    {
        result = 3.14159274 - result;
    }

    if (y < 0)
    {
        result = -result;
    }

    return result;
}