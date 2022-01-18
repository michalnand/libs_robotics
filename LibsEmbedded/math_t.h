#ifndef _MATH_T_H_
#define _MATH_T_H_

template<class DType>
DType abs(DType v)
{
    if (v < 0)
    {
        v = -v;
    }

    return v;
}


template<class DType>
DType min(DType va, DType vb)
{
    if (va < vb)
    {
        return va;
    }
    else
    {
        return vb;
    }
}

template<class DType>
DType max(DType va, DType vb)
{
    if (va > vb)
    {
        return va;
    }
    else
    {
        return vb;
    }
}

template<class DType>
DType sgn(DType v)
{
    if (v > 0)
    {
        return 1.0;
    }
    else
    {
        return -1.0;
    }
}


template<class DType>
DType clamp(DType v, DType min_v, DType max_v)
{
    if (v < min_v)
    {
        v = min_v;
    }

    if (v > max_v)
    {
        v = max_v;
    }

    return v;
}

template<class DType>
DType shrink(DType v, DType s_low, DType s_high)
{
    if (v < s_low || v > s_high)
    {
        return v;
    }
    else
    {
        return 0;
    }
}



#endif