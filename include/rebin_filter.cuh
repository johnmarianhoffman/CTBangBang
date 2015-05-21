#include <recon_structs.h>

#define pi 3.1415926535897f

texture<float,cudaTextureType2D,cudaReadModeElementType> tex_a;
texture<float,cudaTextureType2D,cudaReadModeElementType> tex_b;
texture<float,cudaTextureType2D,cudaReadModeElementType> tex_c;
texture<float,cudaTextureType2D,cudaReadModeElementType> tex_d;

__constant__ struct ct_geom d_cg;
__constant__ struct recon_info d_ri;

/* --- Helper functions (called by kernels) --- */
__device__ inline float angle(float x1,float x2,float y1,float y2){
    return asin((x1*y2-x2*y1)/(sqrt(x1*x1+x2*x2)*sqrt(y1*y1+y2*y2)));
}

__device__ inline float beta_rk(float da,float dr,float channel,int os_flag){
    float b0=(channel-pow(2.0f,os_flag)*d_cg.central_channel)*(d_cg.fan_angle_increment/pow(2.0f,os_flag));
    return angle(-(d_cg.r_f+dr),-(da),-(d_cg.src_to_det*cos(b0)+dr),-(d_cg.src_to_det*sin(b0)+da));
}

__device__ inline float r_fr(float da, float dr){
    return sqrt((d_cg.r_f+dr)*(d_cg.r_f+dr)+da*da);
}

__device__ inline float get_beta_idx(float beta,float * beta_lookup,int n_elements){
    int idx_low=0;

    while (beta>beta_lookup[idx_low]&&idx_low<(n_elements-1)){
    	idx_low++;
    }

    if (idx_low==0)
	idx_low++; 
    
    return (float)idx_low-1.0f+(beta-beta_lookup[idx_low-1])/(beta_lookup[idx_low]-beta_lookup[idx_low-1]);
}

__global__ void beta_lookup(float * lookup,float dr, float da,int os_flag){
    int channel=threadIdx.x+blockIdx.x*blockDim.x;   
    lookup[channel]=beta_rk(da,dr,channel,os_flag);
}

/* --- No flying focal spot rebinning kernels --- */
__global__ void n1_rebin(float * output,int row){
    int channel = threadIdx.x + blockDim.x*blockIdx.x;
    int proj    = threadIdx.y + blockDim.y*blockIdx.y;
    
    float beta=asin((channel-2*d_cg.central_channel)*(d_cg.fan_angle_increment/2));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2.0f*pi);
    float beta_idx=beta/d_cg.fan_angle_increment+d_cg.central_channel;

    output[d_cg.n_channels_oversampled*(d_ri.n_proj_pull/d_ri.n_ffs)*row+(d_ri.n_proj_pull/d_ri.n_ffs)*channel+proj]=tex2D(tex_a,beta_idx+0.5f,alpha_idx+0.5f);
}

__global__ void n2_rebin(float * output,int row){
    int channel = threadIdx.x + blockDim.x*blockIdx.x;
    int proj    = threadIdx.y + blockDim.y*blockIdx.y;

    float beta=asin((channel-2*d_cg.central_channel)*(d_cg.fan_angle_increment/2));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2*pi);
    float beta_idx=beta/d_cg.fan_angle_increment+d_cg.central_channel;

    output[d_cg.n_channels_oversampled*(d_ri.n_proj_pull/d_ri.n_ffs)*row+(d_ri.n_proj_pull/d_ri.n_ffs)*channel+proj]=tex2D(tex_b,beta_idx+0.5f,alpha_idx+0.5f);
}

/* --- Phi only flying focal spot rebinning kernels ---*/
__global__ void p_reshape(float * raw, float * out,int offset){
    int channel=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int proj=blockIdx.z;

    int idx_out=(proj+offset/2)+(d_ri.n_proj_pull/d_ri.n_ffs*channel)+(d_ri.n_proj_pull/d_ri.n_ffs*d_cg.n_channels*row);
    int idx_out_offset=idx_out+(d_cg.n_rows_raw*d_cg.n_channels*d_ri.n_proj_pull/d_ri.n_ffs);
    
    out[idx_out]       =raw[d_cg.n_channels*d_cg.n_rows_raw *(2*proj) +d_cg.n_channels*row+channel];
    out[idx_out_offset]=raw[d_cg.n_channels*d_cg.n_rows_raw*(2*proj+1)+d_cg.n_channels*row+channel];
}

__global__ void p1_rebin_t(float * output,float da,int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = d_cg.n_channels_oversampled*n_proj*row+n_proj*(2*channel)+proj;

    float beta = asin((channel-d_cg.central_channel)*d_cg.fan_angle_increment*d_cg.r_f/r_fr(da,0.0));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2.0f*pi);
    
    output[out_idx]=tex2D(tex_a,alpha_idx+0.5f,channel+0.5f);
}

__global__ void p2_rebin_t(float * output,float da,int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = d_cg.n_channels_oversampled*n_proj*row+n_proj*(2*channel+1)+proj;

    float beta = asin((channel-d_cg.central_channel)*d_cg.fan_angle_increment*d_cg.r_f/r_fr(-da,0.0));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2.0f*pi);
    
    output[out_idx]=tex2D(tex_b,alpha_idx+0.5f,channel+0.5f);
}

__global__ void p1_rebin(float* output,float da,int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = d_cg.n_channels_oversampled*n_proj*row+n_proj*channel+proj;

    float beta  = asin((channel-2*d_cg.central_channel)*(d_cg.fan_angle_increment/2));
    //float beta = asin((channel-2*d_cg.central_channel)*(d_cg.fan_angle_increment/2)*d_cg.r_f/r_fr(da,0.0));
    float beta_idx=beta/(d_cg.fan_angle_increment/2.0f)+2.0f*d_cg.central_channel;

    output[out_idx]=tex2D(tex_a,proj+0.5f,beta_idx+0.5f); 
	
}

__global__ void p2_rebin(float* output,float da,int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = d_cg.n_channels_oversampled*n_proj*row+n_proj*channel+proj;

    float beta  = asin((channel-2*d_cg.central_channel)*(d_cg.fan_angle_increment/2));
    //float beta = asin((channel-2*d_cg.central_channel)*(d_cg.fan_angle_increment/2)*d_cg.r_f/r_fr(-da,0.0));
    float beta_idx=beta/(d_cg.fan_angle_increment/2.0f)+2.0f*d_cg.central_channel;

    output[out_idx]=tex2D(tex_b,proj+0.5f,beta_idx+0.5f);     
}


/* --- Z only flying focal spot rebinning kernels ---*/
__global__ void z_reshape(float * raw, float * out,int offset){
    int channel=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int proj=blockIdx.z;

    int idx_out=(proj+offset/2)+(d_ri.n_proj_pull/d_ri.n_ffs*channel)+(d_ri.n_proj_pull/d_ri.n_ffs*d_cg.n_channels*row);
    int idx_out_offset=idx_out+(d_cg.n_rows_raw*d_cg.n_channels*d_ri.n_proj_pull/d_ri.n_ffs);
    
    out[idx_out]       =raw[d_cg.n_channels*d_cg.n_rows_raw *(2*proj) +d_cg.n_channels*row+channel];
    out[idx_out_offset]=raw[d_cg.n_channels*d_cg.n_rows_raw*(2*proj+1)+d_cg.n_channels*row+channel];
}


__global__ void z1_rebin(float * output,float * beta_lookup,float dr,int row){
    // This kernel handles all projections coming from focal spot 1
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    float beta=asin((channel-2.0f*d_cg.central_channel)*(d_cg.fan_angle_increment/2.0f)*d_cg.r_f/r_fr(0.0f,-dr));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2.0f*pi);
    float beta_idx=get_beta_idx(beta,beta_lookup,d_cg.n_channels);
    
    __syncthreads();

    output[(d_ri.n_proj_pull/d_ri.n_ffs)*d_cg.n_channels_oversampled*2*row+(d_ri.n_proj_pull/d_ri.n_ffs)*channel+proj]=tex2D(tex_a,alpha_idx+0.5f,beta_idx+0.5f);
}

__global__ void z2_rebin(float * output,float * beta_lookup,float dr,int row){
    // This kernel handles all projections coming from focal spot 2
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    float beta=asin((channel-2.0f*d_cg.central_channel)*(d_cg.fan_angle_increment/2.0f)*d_cg.r_f/r_fr(0.0f,dr));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2*pi);
    float beta_idx=get_beta_idx(beta,beta_lookup,d_cg.n_channels);
    
    __syncthreads();

    output[(d_ri.n_proj_pull/d_ri.n_ffs)*d_cg.n_channels_oversampled*(2*row+1)+(d_ri.n_proj_pull/d_ri.n_ffs)*channel+proj]=tex2D(tex_b,alpha_idx+0.5f,beta_idx+0.5f);
}

/* --- Z only flying focal spot rebinning kernels ---*/
__global__ void a_reshape(float * raw, float * out,int offset){
    int channel=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int proj=blockIdx.z;

    int idx_out_1=(proj+offset/4)+(d_ri.n_proj_pull/d_ri.n_ffs*channel)+(d_ri.n_proj_pull/d_ri.n_ffs*d_cg.n_channels*row);
    int idx_out_2=idx_out_1+(d_cg.n_rows_raw*d_cg.n_channels*d_ri.n_proj_pull/d_ri.n_ffs);
    int idx_out_3=idx_out_2+(d_cg.n_rows_raw*d_cg.n_channels*d_ri.n_proj_pull/d_ri.n_ffs);
    int idx_out_4=idx_out_3+(d_cg.n_rows_raw*d_cg.n_channels*d_ri.n_proj_pull/d_ri.n_ffs);    
    
    out[idx_out_1]=raw[d_cg.n_channels*d_cg.n_rows_raw *(4*proj) +d_cg.n_channels*row+channel];
    out[idx_out_2]=raw[d_cg.n_channels*d_cg.n_rows_raw*(4*proj+1)+d_cg.n_channels*row+channel];
    out[idx_out_3]=raw[d_cg.n_channels*d_cg.n_rows_raw*(4*proj+2)+d_cg.n_channels*row+channel];
    out[idx_out_4]=raw[d_cg.n_channels*d_cg.n_rows_raw*(4*proj+3)+d_cg.n_channels*row+channel];
}

__global__ void a1_rebin_t(float * output,float da, float dr, int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = d_cg.n_channels_oversampled*n_proj*row+n_proj*(2*channel)+proj;

    float beta = asin((channel-d_cg.central_channel)*d_cg.fan_angle_increment*d_cg.r_f/r_fr(da,-dr));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2.0f*pi);
    
    output[out_idx]=tex2D(tex_a,alpha_idx+0.5f,channel+0.5f);
}

__global__ void a2_rebin_t(float * output,float da, float dr, int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = d_cg.n_channels_oversampled*n_proj*row+n_proj*(2*channel+1)+proj;

    float beta = asin((channel-d_cg.central_channel)*d_cg.fan_angle_increment*d_cg.r_f/r_fr(-da,-dr));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2.0f*pi);
    
    output[out_idx]=tex2D(tex_b,alpha_idx+0.5f,channel+0.5f);
}

__global__ void a3_rebin_t(float * output,float da, float dr, int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = d_cg.n_channels_oversampled*n_proj*row+n_proj*(2*channel)+proj;

    float beta = asin((channel-d_cg.central_channel)*d_cg.fan_angle_increment*d_cg.r_f/r_fr(da,dr));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2.0f*pi);
    
    output[out_idx]=tex2D(tex_c,alpha_idx+0.5f,channel+0.5f);
}

__global__ void a4_rebin_t(float * output,float da, float dr, int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = d_cg.n_channels_oversampled*n_proj*row+n_proj*(2*channel+1)+proj;

    float beta = asin((channel-d_cg.central_channel)*d_cg.fan_angle_increment*d_cg.r_f/r_fr(-da,dr));
    float alpha_idx=(proj)-beta*d_cg.n_proj_turn/(2.0f*pi);
    
    output[out_idx]=tex2D(tex_d,alpha_idx+0.5f,channel+0.5f);
}

__global__ void a1_rebin_b(float * output,float * beta_lookup,float dr,int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = n_proj*d_cg.n_channels_oversampled*2*row+n_proj*channel+proj;
   
    float beta=asin((channel-2.0f*d_cg.central_channel)*(d_cg.fan_angle_increment/2.0f)*d_cg.r_f/r_fr(0.0f,-dr));
    float beta_idx=get_beta_idx(beta,beta_lookup,d_cg.n_channels_oversampled);
    
    __syncthreads();

    output[out_idx]=tex2D(tex_a,proj+0.5f,beta_idx+0.5f);
}

__global__ void a2_rebin_b(float * output,float * beta_lookup,float dr,int row){
    int channel = threadIdx.x+blockIdx.x*blockDim.x;
    int proj    = threadIdx.y+blockIdx.y*blockDim.y;

    int n_proj  = d_ri.n_proj_pull/d_ri.n_ffs;
    int out_idx = n_proj*d_cg.n_channels_oversampled*(2*row+1)+n_proj*channel+proj;
    
    float beta=asin((channel-2.0f*d_cg.central_channel)*(d_cg.fan_angle_increment/2.0f)*d_cg.r_f/r_fr(0.0f,dr));
    float beta_idx=get_beta_idx(beta,beta_lookup,d_cg.n_channels_oversampled);
    
    __syncthreads();

    output[out_idx]=tex2D(tex_b,proj+0.5f,beta_idx+0.5f);
}


/* --- Filter kernel (we've gotta make this better somehow...) --- */
__global__ void filter(float * output, float * filter, int row){

    int proj=threadIdx.x+blockDim.x*blockIdx.x;

    float r[1500];
    float f[3000];
    float Result[(4500)-1];

    for (int i=0;i<2*d_cg.n_channels_oversampled;i++){
	f[i]=filter[i];
    }

    for (int i=0;i<d_cg.n_channels_oversampled;i++){
	r[i]=output[d_cg.n_channels_oversampled*(d_ri.n_proj_pull/d_ri.n_ffs)*row+(d_ri.n_proj_pull/d_ri.n_ffs)*i+proj];
    }
    
    size_t kmin, kmax,count;
    size_t s_length=d_cg.n_channels_oversampled;
    size_t k_length=2*d_cg.n_channels_oversampled;

    //size_t l=d_cg.n_channels_oversampled;
    
    for (int n=k_length/2;n<((k_length+s_length)-k_length/2);n++){
	Result[n]=0;
	kmin = (n >= k_length - 1) ? n - (k_length - 1) : 0;
	kmax = (n < s_length - 1) ? n : s_length - 1;
	for (count = kmin; count <= kmax; count++){
	    Result[n] += r[count] * f[n - count];
	}
    }

    for (int i=0;i<d_cg.n_channels_oversampled;i++){
	output[d_cg.n_channels_oversampled*(d_ri.n_proj_pull/d_ri.n_ffs)*row+(d_ri.n_proj_pull/d_ri.n_ffs)*i+proj]=Result[k_length/2+i];
    }
}

