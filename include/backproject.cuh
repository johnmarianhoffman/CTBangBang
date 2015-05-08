#include <recon_structs.h>

#define pi 3.1415368979f
#define K 1
#define I 32

texture<float,cudaTextureType2D,cudaReadModeElementType> tex_a;
texture<float,cudaTextureType2D,cudaReadModeElementType> tex_b;

__constant__ struct ct_geom d_cg;
__constant__ struct recon_params d_rp;

__device__ inline float W(float q){
    float out;
    float Q=0.6f;
    if (fabs(q)<Q){
	out=1.0f;
    }
    else if ((fabs(q)>=Q)&&(fabs(q)<1.0f)){
	out=pow(cos((pi/2.0f)*(fabs(q)-Q)/(1.0f-Q)),2.0f);
    }
    else {
	out=0.0f;
    }
    return out;
}

__global__ void bp_a(float * output,int proj_idx,float tube_start,int n_half_turns){
    //Initialize sum voxels and weight voxels
    float s[K]={0};
    float s_t[K]={0};
    float h_t[K]={0};
    
    // Get xyz indices, convert to spatial coordinates
    int xi=threadIdx.x+blockIdx.x*blockDim.x;
    int yi=threadIdx.y+blockIdx.y*blockDim.y;
    int zi=K*(threadIdx.z+blockIdx.z*blockDim.z);
    
    float x=(d_rp.recon_fov/d_rp.nx)*((float)xi-(d_rp.nx-1)/2.0f)+d_rp.x_origin;
    float y=(d_rp.recon_fov/d_rp.ny)*((float)yi-(d_rp.ny-1)/2.0f)+d_rp.y_origin;
    float z=zi*d_rp.slice_thickness+d_cg.z_rot/2.0f+d_cg.z_rot*tube_start/(2.0f*pi);

    for (int i=0;i<I;i++){

	for (int ii=0;ii<K;ii++){ // zero out our holder arrays
	    s_t[ii]=0.0f;
	    h_t[ii]=0.0f;
	}
	
	for (int k=0;k<n_half_turns;k++){
	    float theta=tube_start+(2.0f*pi/d_cg.n_proj_turn)*(proj_idx+i)+k*pi;
	    float phat=x*sin(theta)-y*cos(theta);
	    float p_idx=phat/(d_cg.r_f*d_cg.fan_angle_increment/2.0f)+2.0f*d_cg.central_channel;
	    
	    for (int j=0;j<K;j++){
		z+=j*d_rp.slice_thickness;
	    
		float ray_pos=(d_cg.z_rot*(theta-asin(phat/d_cg.r_f))/(2.0f*pi));
		float lhat=sqrt(pow(d_cg.r_f,2.0f)-pow(phat,2.0f))-x*cos(theta)-y*sin(theta);
		float qhat=(z-ray_pos)/(lhat*tan(d_cg.theta_cone/2.0f));
		float q_idx=((qhat+1.0f)/2.0f)*(d_cg.n_rows-1.0f)+d_cg.n_rows*i+k*I*d_cg.n_rows;

		if (fabs(qhat)<=1){
		    s_t[j]+=tex2D(tex_a,p_idx+0.5,q_idx+0.5)*W(qhat);
		    h_t[j]+=W(qhat);
		}
		__syncthreads();
	    
	    }	    
	}
	
	for (int kk=0;kk<K;kk++){
	    if (h_t[kk]!=0){
		s[kk]+=(1.0f/h_t[kk])*s_t[kk];
	    }
	}
    }
    
    __syncthreads();

    for (int k=0;k<K;k++){
	output[d_rp.nx*d_rp.ny*(zi+k)+d_rp.nx*xi+yi]+=s[k];//(1.0f/h[k])*s[k];
    }
}

__global__ void bp_b(float * output,int proj_idx,float tube_start,int n_half_turns){
    //Initialize sum voxels and weight voxels
    float s[K]={0};
    float s_t[K]={0};
    float h_t[K]={0};
    
    // Get xyz indices, convert to spatial coordinates
    int xi=threadIdx.x+blockIdx.x*blockDim.x;
    int yi=threadIdx.y+blockIdx.y*blockDim.y;
    int zi=K*(threadIdx.z+blockIdx.z*blockDim.z);
    
    float x=(d_rp.recon_fov/d_rp.nx)*((float)xi-(d_rp.nx-1)/2.0f)+d_rp.x_origin;
    float y=(d_rp.recon_fov/d_rp.ny)*((float)yi-(d_rp.ny-1)/2.0f)+d_rp.y_origin;
    float z=zi*d_rp.slice_thickness+d_cg.z_rot/2.0f+d_cg.z_rot*tube_start/(2.0f*pi);

    for (int i=0;i<I;i++){
	
	for (int ii=0;ii<K;ii++){ // zero out our holder arrays
	    s_t[ii]=0.0f;
	    h_t[ii]=0.0f;
	}
	
	for (int k=0;k<n_half_turns;k++){
	    float theta=tube_start+(2.0f*pi/d_cg.n_proj_turn)*(proj_idx+i)+k*pi;
	    float phat=x*sin(theta)-y*cos(theta);
	    float p_idx=phat/(d_cg.r_f*d_cg.fan_angle_increment/2.0f)+2.0f*d_cg.central_channel;

	    for (int j=0;j<K;j++){
		z+=j*d_rp.slice_thickness;
	    
		float ray_pos=(d_cg.z_rot*(theta-asin(phat/d_cg.r_f))/(2.0f*pi));
		float lhat=sqrt(pow(d_cg.r_f,2.0f)-pow(phat,2.0f))-x*cos(theta)-y*sin(theta);
		float qhat=(z-ray_pos)/(lhat*tan(d_cg.theta_cone/2.0f));
		float q_idx=((qhat+1.0f)/2.0f)*(d_cg.n_rows-1.0f)+d_cg.n_rows*i+k*I*d_cg.n_rows;

		if (fabs(qhat)<=1){
		    s_t[j]+=tex2D(tex_b,p_idx+0.5,q_idx+0.5)*W(qhat);
		    h_t[j]+=W(qhat);
		}
		__syncthreads();
	    }
	}

	for (int kk=0;kk<K;kk++){
	    if (h_t[kk]!=0){
		s[kk]+=(1.0f/h_t[kk])*s_t[kk];
	    }
	}
    }

    __syncthreads();
    
    for (int k=0;k<K;k++){
	output[d_rp.nx*d_rp.ny*(zi+k)+d_rp.nx*xi+yi]+=s[k];//(1.0f/h[k])*s[k];
    }
}


