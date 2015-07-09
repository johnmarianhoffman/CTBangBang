#ifndef interp_h
#define interp_h

struct array_dims{
    int idx1;
    int idx2;
    int idx3;
};

float interp1(float * array, float idx1);
float interp2(float * array, struct array_dims dim, float idx1,float idx2);
float interp3(float * array, struct array_dims dim, float idx1,float idx2,float idx3);

#endif
