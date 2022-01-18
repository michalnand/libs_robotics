#include <iostream>
#include <math.h>
#include <fmath.h>

void sqrt_test(float min, float max, float step)
{
  std::cout <<"sqrt test\n";

  float error_max = 0;

  float error_mean = 0;
  float count      = 0;

  for (float x = min; x < max; x+= step)
  {
    float y_ref = sqrt(x);
    float y_test = fsqrt(x);

    float diff = fabs(y_ref - y_test);
    float error = diff*100.0/(fabs(y_ref) + 10e-30);

    if (error > error_max)
    {
      error_max = error;
    }

    error_mean+= error;
    count++;

    
    if (error > 0.0001)
    {
      std::cout << "ERROR\n";
      std::cout << x << " " <<  y_ref << " " << y_test << " " << error << "\n";

      exit(EXIT_FAILURE);
    }
  }

  std::cout << "error max = " << error_max << "[%]\n";
  std::cout << "error mean = " << error_mean/count << "[%]\n\n";
}


void sin_test(float min, float max, float step)
{
  std::cout <<"sin test\n";

  float error_max = 0;

  float error_mean = 0;
  float count      = 0;

  for (float x = min; x < max; x+= step)
  {
    float y_ref  = sin(x);
    float y_test = fsin(x);

    float diff  = fabs(y_ref - y_test);
    float error = diff*100.0/(fabs(y_ref) + 10e-30);

    if (error > error_max)
    {
      error_max = error;
    }

    error_mean+= error;
    count++;

    if (error > 0.001 && diff > 0.0001)
    {
      std::cout << "ERROR\n";
      std::cout << x << " " << y_ref << " " << y_test << " " << error << "\n";

      exit(EXIT_FAILURE);
    }
  }

  std::cout << "error max = " << error_max << "[%]\n";
  std::cout << "error mean = " << error_mean/count << "[%]\n\n";
}


void cos_test(float min, float max, float step)
{
  std::cout <<"cos test\n";

  float error_max = 0;

  float error_mean = 0;
  float count      = 0;

  for (float x = min; x < max; x+= step)
  {
    float y_ref  = cos(x);
    float y_test = fcos(x);

    float diff = y_ref - y_test;
    if (diff < 0)
    {
      diff = -diff;
    }

    float error = diff*100.0/(fabs(y_ref) + 10e-30);

    if (error > error_max)
    {
      error_max = error;
    }

    error_mean+= error;
    count++;

    if (error > 0.001 && diff > 0.0001)
    {
      std::cout << "ERROR\n";
      std::cout << x << " " << y_ref << " " << y_test << " " << error << "\n";

      exit(EXIT_FAILURE);
    }
  }

  std::cout << "error max = " << error_max << "[%]\n";
  std::cout << "error mean = " << error_mean/count << "[%]\n\n";
}


void tan_test(float min, float max, float step)
{
  std::cout <<"tan test\n";

  float error_max = 0;

  float error_mean = 0;
  float count      = 0;

  for (float x = min; x < max; x+= step)
  {
    float y_ref  = tan(x);
    float y_test = ftan(x);

    float diff = y_ref - y_test;
    if (diff < 0)
    {
      diff = -diff;
    }

    float error = diff*100.0/(fabs(y_ref) + 10e-30);

    if (error > error_max)
    {
      error_max = error;
    }

    error_mean+= error;
    count++;

    if (error > 0.01 && diff > 0.0001)
    {
      std::cout << "ERROR\n";
      std::cout << x << " " << y_ref << " " << y_test << " " << error << "\n";

      exit(EXIT_FAILURE);
    }
  }

  std::cout << "error max = " << error_max << "[%]\n";
  std::cout << "error mean = " << error_mean/count << "[%]\n\n";
}


void asin_test(float min, float max, float step)
{
  std::cout <<"asin test\n";

  float error_max = 0;

  float error_mean = 0;
  float count      = 0;

  for (float x = min; x < max; x+= step)
  { 
    float y_ref  = asin(x);
    float y_test = fasin(x);

    float diff = fabs(y_ref - y_test);
    float error = diff*100.0/(fabs(y_ref) + 10e-30);

    if (error > error_max)
    {
      error_max = error;
    }

    error_mean+= error;
    count++;

    if (error > 0.5)
    {
      std::cout << "ERROR\n";
      std::cout << x << " " << y_ref << " " << y_test << " " << error << "\n";

      exit(EXIT_FAILURE);
    }
  }

  std::cout << "error max = " << error_max << "[%]\n";
  std::cout << "error mean = " << error_mean/count << "[%]\n\n";
}


void acos_test(float min, float max, float step)
{
  std::cout <<"acos test\n";

  float error_max = 0;

  float error_mean = 0;
  float count      = 0;

  for (float x = min; x < max; x+= step)
  { 
    float y_ref  = acos(x);
    float y_test = facos(x);

    float diff = fabs(y_ref - y_test);
    float error = diff*100.0/(fabs(y_ref) + 10e-30);

    if (error > error_max)
    {
      error_max = error;
    }

    error_mean+= error;
    count++;

    if (error > 0.5)
    {
      std::cout << "ERROR\n";
      std::cout << x << " " << y_ref << " " << y_test << " " << error << "\n";

      exit(EXIT_FAILURE);
    }
  }

  std::cout << "error max = " << error_max << "[%]\n";
  std::cout << "error mean = " << error_mean/count << "[%]\n\n";
}

void atan_test(float min, float max, float step)
{
  std::cout <<"atan test\n";

  float error_max = 0;

  float error_mean = 0;
  float count      = 0;

  for (float x = min; x < max; x+= step)
  { 
    float y_ref  = atan(x);
    float y_test = fatan(x);

    float diff = fabs(y_ref - y_test);
    float error = diff*100.0/(fabs(y_ref) + 10e-30);

    if (error > error_max)
    {
      error_max = error;
    }

    error_mean+= error;
    count++;

    if (error > 0.5)
    {
      std::cout << "ERROR\n";
      std::cout << x << " " << y_ref << " " << y_test << " " << error << "\n";

      exit(EXIT_FAILURE);
    }
  }

  std::cout << "error max = " << error_max << "[%]\n";
  std::cout << "error mean = " << error_mean/count << "[%]\n\n";
}

int main() 
{ 
    sqrt_test(0, 65536, 0.1);
    sin_test(-8.0*PI, 8.0*PI, 0.0001);
    cos_test(-8.0*PI, 8.0*PI, 0.0001);
    tan_test(-0.999*PI/2.0, 0.999*PI/2.0, 0.0001);

    asin_test(-1.0, 1.0, 0.0001);
    acos_test(-1.0, 1.0, 0.0001);
    atan_test(-10000.0, 10000.0, 0.001);
}
