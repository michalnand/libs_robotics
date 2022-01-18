#include "imu.h"


IMU::IMU()
{
    result.x = 0;
    result.y = 0;
    result.z = 0;

    this->alpha = 0.05;
}

IMU::~IMU()
{

}

void IMU::init(float alpha)
{
    result.x = 0;
    result.y = 0;
    result.z = 0;

    this->alpha = alpha;
}

Vect3d<float> IMU::step(float ax, float ay, float az, float gx, float gy, float gz, float dt)
{
    Vect3d<float> acc_result;

    ax = clamp(ax, -G_const, G_const);
    ay = clamp(ay, -G_const, G_const);
    az = clamp(az, -G_const, G_const);

    acc_result.x = -fatan2(az, ay);      //roll
    acc_result.y = fasin(ax/G_const);    //pitch
    acc_result.z = 0;                    //yaw
    
  
    gx = -shrink(gx, -0.01f, 0.01f);
    gy = -shrink(gy, -0.01f, 0.01f);
    gz = shrink(gz, -0.01f, 0.01f);

    
    result.x = alpha*acc_result.x + (1.0 - alpha)*(result.x + gx*dt);
    result.y = alpha*acc_result.y + (1.0 - alpha)*(result.y + gy*dt);
    result.z = result.z + gz*dt;
    

    return result;
}
