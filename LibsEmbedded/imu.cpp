#include "imu.h"

#define G_const     (float)9.81

IMU::IMU()
{
    result.x = 0;
    result.y = 0;
    result.z = 0;

    this->alpha = 0.05;
    this->dt    = 0.01;
}

IMU::~IMU()
{

}

void IMU::init(float alpha, float dt)
{
    result.x = 0;
    result.y = 0;
    result.z = 0;

    this->alpha = alpha;
    this->dt    = dt;
}

Vect3d<float> IMU::step(float ax, float ay, float az, float gx, float gy, float gz)
{
    Vect3d<float> acc_result;

    ax = clamp(ax, -G_const, G_const);
    ay = clamp(ay, -G_const, G_const);
    az = clamp(az, -G_const, G_const);


    /*
    acc_result.x = atan2(ay, az);       //roll
    acc_result.y = asin(ax/G_const);    //pitch
    acc_result.z = 0;                   //yaw

    auto gyro_result = body_rates_to_euler_rates(result.x, result.y, gx, gy, gz);


    result.x = alpha*acc_result.x + (1.0 - alpha)*(result.x + gyro_result.x*dt);
    result.y = alpha*acc_result.y + (1.0 - alpha)*(result.y + gyro_result.y*dt);
    
    //result.z = alpha*acc_result.z + (1.0 - alpha)*(result.z + gyro_result.z*dt)
    result.z = result.z + gyro_result.z*dt;
    */
 
    acc_result.x = atan2(ay, az);       //roll
    acc_result.y = asin(ax/G_const);    //pitch
 
    result.x = alpha*acc_result.x + (1.0 - alpha)*(result.x);
    result.y = alpha*acc_result.y + (1.0 - alpha)*(result.y);
    

    return result;
}


Vect3d<float> IMU::body_rates_to_euler_rates(float phi, float theta, float p, float q, float r)
{
    Vect3d<float> result;

    float sec_theta = 1.0/(cos(theta));
    float sin_phi   = sin(phi);
    float cos_phi   = cos(phi);
    float tan_theta = tan(theta);

    result.x = 1.0*p    +   sin_phi*tan_theta*q    +   cos_phi*tan_theta*r;
    result.y =              cos_phi*q              -   sin_phi*r;
    result.z =              sin_phi*sec_theta*q    +   cos_phi*sec_theta*r;

    return result;
}