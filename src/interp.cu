#include <math.h>
#include <interp.h>

float interp1(float * array, float idx1){
    float w=idx1-floor(idx1);
    return array[(int)idx1]*(1.0f-w)+array[(int)idx1+1]*w;
}

float interp2(float * array, struct array_dims dim, float idx1,float idx2){
    //Assumes idx1 is stored linearly in memory with stride 1
    //idx2 is stored with stride dim_idx 1
    //
    //     dim1------->
    //dim2  0   0   0   0   0   0   0   0
    //|     1   1   1   1   1   1   1   1
    //|     2   2   2   2   2   2   2   2 
    //|     .           .
    //|     .               .
    //|     .                   .

    float val;

    // Edge mode "Border"
    // i.e. if user requests data outside of array, return 0; 
    if ( idx1>(dim.idx1-2) || (idx1<0) || idx2>(dim.idx2-2) || (idx2<0) )
	val=0.0f;
    else {
    float v=idx1-floor(idx1);
    float w=idx2-floor(idx2);

    int a_idx1=(int)floor(idx2)*dim.idx1+(int)floor(idx1);
    int a_idx2=(int)floor(idx2)*dim.idx1+(int)floor(idx1)+1;
    int a_idx3=((int)floor(idx2)+1)*dim.idx1+(int)floor(idx1);
    int a_idx4=((int)floor(idx2)+1)*dim.idx1+(int)floor(idx1)+1;

    val=   array[a_idx1]  *  (1.0f-v)  *(1.0f-w)
	+  array[a_idx2]  *      v     *(1.0f-w)
	+  array[a_idx3]  *  (1.0f-v)  *    w
	+  array[a_idx4]  *      v     *    w;

    }
    
    return val;
}

float interp3(float * array, struct array_dims dim, float idx1,float idx2,float idx3){
    // dim.idx1 and idx1 are stored linearly in memory
    // dim.idx2 and idx2 are stored in memory with stride dim.idx1
    // dim.idx3 and idx3 are stored in memory with stride dim.idx1*dim.idx2
    
    // Clamping
    if (idx1>(dim.idx1-2))
	idx1=dim.idx1-2.0f;
    if (idx1<0)
	idx1=0.0f;
    if (idx2>(dim.idx2-2))
	idx2=dim.idx2-2.0f;
    if (idx2<0)
	idx2=0.0f;
    if (idx3>(dim.idx3-2))
	idx3=dim.idx3-2.0f;
    if (idx3<0)
	idx3=0.0f;

    // Find weights
    float u=idx1-floor(idx1);
    float v=idx2-floor(idx2);
    float w=idx3-floor(idx3);

    // Find linear indices for interpolation points
    int a_idx1=(int)floor(idx3)*dim.idx2*dim.idx1     + (int)floor(idx2)*dim.idx1     + (int)floor(idx1);
    int a_idx2=(int)floor(idx3)*dim.idx2*dim.idx1     + (int)floor(idx2)*dim.idx1     + (int)floor(idx1) + 1;
    int a_idx3=(int)floor(idx3)*dim.idx2*dim.idx1     + ((int)floor(idx2)+1)*dim.idx1 + (int)floor(idx1);
    int a_idx4=(int)floor(idx3)*dim.idx2*dim.idx1     + ((int)floor(idx2)+1)*dim.idx1 + (int)floor(idx1) + 1;    
    int a_idx5=((int)floor(idx3)+1)*dim.idx2*dim.idx1 + (int)floor(idx2)*dim.idx1     + (int)floor(idx1);
    int a_idx6=((int)floor(idx3)+1)*dim.idx2*dim.idx1 + (int)floor(idx2)*dim.idx1     + (int)floor(idx1) + 1;
    int a_idx7=((int)floor(idx3)+1)*dim.idx2*dim.idx1 + ((int)floor(idx2)+1)*dim.idx1 + (int)floor(idx1);
    int a_idx8=((int)floor(idx3)+1)*dim.idx2*dim.idx1 + ((int)floor(idx2)+1)*dim.idx1 + (int)floor(idx1) + 1;    

    //Return the interpolation
    return array[a_idx1]  *  (1-u) * (1-v) * (1-w) + 
	   array[a_idx2]  *    u   * (1-v) * (1-w) +
	   array[a_idx3]  *  (1-u) *   v   * (1-w) +
	   array[a_idx4]  *    u   *   v   * (1-w) +
	   array[a_idx5]  *  (1-u) * (1-v) *   w   +
	   array[a_idx6]  *    u   * (1-v) *   w   +
	   array[a_idx7]  *  (1-u) *   v   *   w   +
           array[a_idx8]  *    u   *   v   *   w   ;
}
