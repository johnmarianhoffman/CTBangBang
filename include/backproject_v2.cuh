/* CTBangBang is GPU and CPU CT reconstruction Software */
/* Copyright (C) 2015  John Hoffman */

/* This program is free software; you can redistribute it and/or */
/* modify it under the terms of the GNU General Public License */
/* as published by the Free Software Foundation; either version 2 */
/* of the License, or (at your option) any later version. */

/* This program is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the */
/* GNU General Public License for more details. */

/* You should have received a copy of the GNU General Public License */
/* along with this program; if not, write to the Free Software */
/* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA. */

/* Questions and comments should be directed to */
/* jmhoffman@mednet.ucla.edu with "CTBANGBANG" in the subject line*/

#include <recon_structs.h>
#include <backproject_meta.h>

//#define __DEBUG__

__device__ inline float z_ray_f(float theta, float phat, float l,float q){
    return (d_cg.z_rot/(2.0f*pi))*(theta-asinf(phat/d_cg.r_f))+q*l*tanf(d_cg.theta_cone/2.0f);
}
__device__ inline float lhat_f(float phat,float x, float y, float theta){
    return sqrtf(d_cg.r_f*d_cg.r_f-phat*phat)-x*cosf(theta)-y*sinf(theta);
}
__device__ inline float h_f(float d_z){
    return fmaxf(0.0f,1.0f-fabsf(d_z)/d_rp.coll_slicewidth);
}
__device__ inline float product_h_dxyz_W(float theta,float phat, float x, float y,float z,float q){
    float lhat=lhat_f(phat,x,y,theta);
    float zray_of_lhat=z_ray_f(theta,phat,lhat,q);
    return h_f(zray_of_lhat-z)*W(q);
}

__global__ void bp_2(float * output,int proj_idx,float tube_start,int n_half_turns,int kernel_idx){
    // Get xyz indices, convert to spatial coordinates
    int xi=threadIdx.x+blockIdx.x*blockDim.x;
    int yi=threadIdx.y+blockIdx.y*blockDim.y;
    int zi=K*(threadIdx.z+blockIdx.z*blockDim.z);

    float x;
    if (d_cg.table_direction==-1)
        x=(d_rp.recon_fov/d_rp.nx)*((float)xi-(d_rp.nx-1)/2.0f)+d_rp.x_origin;
    else
	x=(d_rp.recon_fov/d_rp.nx)*(-(float)xi+(d_rp.nx-1)/2.0f)+d_rp.x_origin;
    float y=(d_rp.recon_fov/d_rp.ny)*((float)yi-(d_rp.ny-1)/2.0f)+d_rp.y_origin;    
    float z=zi*d_rp.coll_slicewidth+d_cg.z_rot/2.0f+d_cg.z_rot*tube_start/(2.0f*pi);

    // We need to track the following sums
    // Voxel  sum=h(dxyz)*W*projection data
    // Weight sum=h(dxyz)*W
    float final_voxel=0.0f;
    float voxel=0.0f;
    float weight=0.0f;
    
    if (1){ // for debugging
        for (int i=0;i<I;i++){

            voxel=0.0f;
            weight=0.0f;

            for (int k=0;k<n_half_turns;k++){
        
                float theta=tube_start+(2.0f*pi/d_cg.n_proj_turn)*(proj_idx+i)+k*pi;
                float phat=x*sinf(theta)-y*cosf(theta);
                float p_idx=phat/(d_cg.r_f*d_cg.fan_angle_increment/2.0f)+2.0f*d_cg.central_channel;

                for (int q_idx=0;q_idx<d_cg.n_rows;q_idx++){

                    float q=(-1.0f+(float)q_idx*2.0f/((float)d_cg.n_rows-1.0f));
                    
                    float tmp_sum=product_h_dxyz_W(theta,phat,x,y,z,q);

                    if (kernel_idx==0)
                        voxel+=tmp_sum*tex2D(tex_a,p_idx+0.5f,(float)q_idx+0.5f);
                    else
                        voxel+=0.0f;//tmp_sum*tex2D(tex_b,p_idx+0.5f,(float)q_idx+0.5f);
                    
                    weight+=tmp_sum;
                }
            }

            if (weight!=0)
                final_voxel+=voxel/weight;
        }
    }

    if (sqrtf(x*x+y*y)<(d_rp.acq_fov/2.0)){
        output[d_rp.nx*d_rp.ny*(zi)+d_rp.nx*xi+yi]+= final_voxel*2.0f*pi/(float)d_cg.n_proj_turn;
    }
    else{
        output[d_rp.nx*d_rp.ny*(zi)+d_rp.nx*xi+yi]=0.0f;
    }

}

//__global__ void bp_2_a(float * output,int proj_idx,float tube_start,int n_half_turns){
//    // Get xyz indices, convert to spatial coordinates
//    int xi=threadIdx.x+blockIdx.x*blockDim.x;
//    int yi=threadIdx.y+blockIdx.y*blockDim.y;
//    int zi=K*(threadIdx.z+blockIdx.z*blockDim.z);
//
//    float x;
//    if (d_cg.table_direction==-1)
//        x=(d_rp.recon_fov/d_rp.nx)*((float)xi-(d_rp.nx-1)/2.0f)+d_rp.x_origin;
//    else
//	x=(d_rp.recon_fov/d_rp.nx)*(-(float)xi+(d_rp.nx-1)/2.0f)+d_rp.x_origin;
//    float y=(d_rp.recon_fov/d_rp.ny)*((float)yi-(d_rp.ny-1)/2.0f)+d_rp.y_origin;    
//    float z=zi*d_rp.coll_slicewidth+d_cg.z_rot/2.0f+d_cg.z_rot*tube_start/(2.0f*pi);
//
//    // We need to track the following sums
//    // Voxel  sum=h(dxyz)*W*projection data
//    // Weight sum=h(dxyz)*W
//
//    float final_voxel=0.0f;
//    float voxel=0.0f;
//    float weight=0.0f;
//
//    if (1){ // for debugging
//    
//        for (int i=0;i<I;i++){
//            for (int k=0;k<n_half_turns;k++){
//        
//                float theta=tube_start+(2.0f*pi/d_cg.n_proj_turn)*(proj_idx+i)+k*pi;
//                float phat=x*sinf(theta)-y*cosf(theta);
//                float p_idx=phat/(d_cg.r_f*d_cg.fan_angle_increment/2.0f)+2.0f*d_cg.central_channel;
//
//                for (int q_idx=0;q_idx<d_cg.n_rows;q_idx++){
//                    float tmp_sum=product_h_dxyz_W(theta,phat,x,y,z,(float)q_idx);
//
//                    voxel+=tmp_sum*tex2D(tex_a,p_idx+0.5f,(float)q_idx+0.5f);
//                    weight+=tmp_sum;
//                }
//            }
//
//            if (weight!=0)
//                final_voxel+=voxel/weight;
//        }
//    }
//
//    if (sqrtf(x*x+y*y)<(d_rp.acq_fov/2.0)){
//        output[d_rp.nx*d_rp.ny*(zi)+d_rp.nx*xi+yi]+= final_voxel*2.0f*pi/(float)d_cg.n_proj_turn;
//    }
//    else{
//        output[d_rp.nx*d_rp.ny*(zi)+d_rp.nx*xi+yi]=0.0f;
//    }
//
//}
//
//__global__ void bp_2_b(float * output,int proj_idx,float tube_start,int n_half_turns){
//    // Get xyz indices, convert to spatial coordinates
//    int xi=threadIdx.x+blockIdx.x*blockDim.x;
//    int yi=threadIdx.y+blockIdx.y*blockDim.y;
//    int zi=K*(threadIdx.z+blockIdx.z*blockDim.z);
//
//    float x;
//    if (d_cg.table_direction==-1)
//        x=(d_rp.recon_fov/d_rp.nx)*((float)xi-(d_rp.nx-1)/2.0f)+d_rp.x_origin;
//    else
//	x=(d_rp.recon_fov/d_rp.nx)*(-(float)xi+(d_rp.nx-1)/2.0f)+d_rp.x_origin;
//    float y=(d_rp.recon_fov/d_rp.ny)*((float)yi-(d_rp.ny-1)/2.0f)+d_rp.y_origin;    
//    float z=zi*d_rp.coll_slicewidth+d_cg.z_rot/2.0f+d_cg.z_rot*tube_start/(2.0f*pi);
//
//    // We need to track the following sums
//    // Voxel  sum=h(dxyz)*W*projection data
//    // Weight sum=h(dxyz)*W
//
//    float final_voxel=0.0f;
//    float voxel=0.0f;
//    float weight=0.0f;
//
//    if (1){ // for debugging
//    
//        for (int i=0;i<I;i++){
//            for (int k=0;k<n_half_turns;k++){
//        
//                float theta=tube_start+(2.0f*pi/d_cg.n_proj_turn)*(proj_idx+i)+k*pi;
//                float phat=x*sinf(theta)-y*cosf(theta);
//                float p_idx=phat/(d_cg.r_f*d_cg.fan_angle_increment/2.0f)+2.0f*d_cg.central_channel;
//
//                for (int q_idx=0;q_idx<d_cg.n_rows;q_idx++){
//                    float tmp_sum=product_h_dxyz_W(theta,phat,x,y,z,(float)q_idx);
//            
//                    voxel+=tmp_sum*tex2D(tex_b,p_idx+0.5f,(float)q_idx+0.5f);
//                    weight+=tmp_sum;
//                }
//            }
//
//            if (weight!=0)
//                final_voxel+=voxel/weight;
//        }
//    }
//
//    if (sqrtf(x*x+y*y)<(d_rp.acq_fov/2.0)){
//        output[d_rp.nx*d_rp.ny*(zi)+d_rp.nx*xi+yi]+= final_voxel*2.0f*pi/(float)d_cg.n_proj_turn;
//    }
//    else{
//        output[d_rp.nx*d_rp.ny*(zi)+d_rp.nx*xi+yi]=0.0f;
//    }
//
//}
