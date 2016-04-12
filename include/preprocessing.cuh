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

__constant__ struct ct_geom d_cg;
__constant__ struct recon_info d_ri;

__global__ void extract_sup(float * raw,float * sup){

    /* load a projection into thread */
    size_t proj_idx=threadIdx.x+blockIdx.x*blockDim.x;
    size_t proj_size=d_cg.n_channels*d_cg.n_rows_raw;
    size_t row_size=d_cg.n_channels;

    float max_val=0.0f;
    
    for (int i=0; i<d_cg.n_channels; i++){
	for (int j=0; j<d_cg.n_rows_raw; j++){
	    max_val=max(max_val,raw[proj_size*proj_idx+j*row_size+i]);
	}
    }

    sup[proj_idx]=max_val;
    
}

__global__ void smooth_sup(float * sup_raw,float * sup_smooth){

    int proj_idx=threadIdx.x+blockIdx.x*blockDim.x;
    
    // 18 degree value pulled from Kachelreiss and Kalendar 2001
    int n_avg=ceil((18.0f/360.0f)*(float)d_cg.n_proj_turn_ffs);

    for (i=0; i<n_avg;i++){

	

    }
    

}

__global__ void eccentricity(float * sup_smooth, float * ecc){}

__global__ void eccentricity_rescale(float * ecc, float * ecc_rescale){}

__global__ void find_thresholds(float * ecc_rescale,float * thresholds){}

__global__ void filter_projections(float * raw,float * thresholds, float * filtered_raw){} 
