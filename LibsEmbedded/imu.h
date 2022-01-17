#ifndef _IMU_H_
#define _IMU_H_

#include "math.h"

class IMU
{
    public:
        IMU();
        virtual ~IMU();
        void init(float alpha, float dt = 0.01);

        //accelerometer input   : ax, ay, az;   [m/s^2] 
        //gyroscope input       : gx, gy, gz;   [rad/s]
        //returns               : x, y, z;      [rad]
        Vect3d<float> step(float ax, float ay, float az, float gx, float gy, float gz);

    public:
        Vect3d<float> result;

    private:
        Vect3d<float> body_rates_to_euler_rates(float phi, float theta, float p, float q, float r);

    private:
        float alpha, dt;
};

#endif