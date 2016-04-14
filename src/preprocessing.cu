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

#include <stdlib.h>
#include <stdio.h>
#include <recon_structs.h>
#include <preprocessing.cuh>
#include <preprocessing.h>
#include <math.h>


void float_debug(float * array, size_t numel, const char * filename){
    FILE * fid=fopen(filename,"w");
    fwrite(array,sizeof(float),numel,fid);
    fclose(fid);
}

/* Adaptive filtering from Kachelreiss and Kalendar 2001 */
int adaptive_filter_kk(struct recon_metadata * mr){

    // Save some typing
    struct ct_geom cg=mr->cg;
    struct recon_info ri=mr->ri;

    // Copy of ct geometry
    cudaMemcpyToSymbol(d_cg,&cg,sizeof(struct ct_geom),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_ri,&ri,sizeof(struct recon_info),0,cudaMemcpyHostToDevice);

    // Determine maximum of each projection
    float * d_sup;
    float * h_sup=(float*)calloc(ri.n_proj_pull,sizeof(float));
    cudaMalloc(&d_sup,ri.n_proj_pull*sizeof(float));

    float *d_raw;
    cudaMalloc(&d_raw,cg.n_channels*cg.n_rows_raw*ri.n_proj_pull*sizeof(float));
    cudaMemcpy(d_raw,mr->ctd.raw,cg.n_channels*cg.n_rows_raw*ri.n_proj_pull*sizeof(float),cudaMemcpyHostToDevice);

    dim3 extract_sup_threads(128,1,1); // we're guaranteed to get an round number of blocks this way
    dim3 extract_sup_blocks(ri.n_proj_pull/128,1,1);

    extract_sup<<<extract_sup_blocks,extract_sup_threads>>>(d_raw,d_sup);

    cudaMemcpy(h_sup,d_sup,ri.n_proj_pull*sizeof(float),cudaMemcpyDeviceToHost);

    float_debug(h_sup,ri.n_proj_pull,"/home/john/Desktop/h_sup.txt");
    
    // Smooth the array of maxima
    float * d_sup_smooth;
    float * h_sup_smooth=(float*)calloc(ri.n_proj_pull,sizeof(float));
    cudaMalloc(&d_sup_smooth,ri.n_proj_pull*sizeof(float));
    
    dim3 smooth_sup_threads(128,1,1);
    dim3 smooth_sup_blocks(ri.n_proj_pull/128,1,1);

    smooth_sup<<<extract_sup_blocks,extract_sup_threads>>>(d_sup,d_sup_smooth);

    cudaMemcpy(h_sup_smooth,d_sup_smooth,ri.n_proj_pull*sizeof(float),cudaMemcpyDeviceToHost);

    float_debug(h_sup_smooth,ri.n_proj_pull,"/home/john/Desktop/h_sup_smooth.txt");

    // Calculate eccentricity as a function of projection idx
    float * d_ecc;
    float * h_ecc=(float*)calloc(ri.n_proj_pull,sizeof(float));
    cudaMalloc(&d_ecc,ri.n_proj_pull*sizeof(float));

    float * d_p_max;
    float * d_p_min;
    cudaMalloc(&d_p_max,ri.n_proj_pull*sizeof(float));
    cudaMalloc(&d_p_min,ri.n_proj_pull*sizeof(float));
    
    dim3 ecc_threads(128,1,1);
    dim3 ecc_blocks(ri.n_proj_pull/128,1,1);

    eccentricity<<<ecc_blocks,ecc_threads>>>(d_sup_smooth,d_ecc,d_p_max,d_p_min);

    cudaMemcpy(h_ecc,d_ecc,ri.n_proj_pull*sizeof(float),cudaMemcpyDeviceToHost);

    float_debug(h_ecc,ri.n_proj_pull,"/home/john/Desktop/h_ecc_trunc.txt");

    // Find thresholds
    float * d_threshold;
    float * h_threshold=(float *)calloc(ri.n_proj_pull,sizeof(float));
    cudaMalloc(&d_threshold,ri.n_proj_pull*sizeof(float));

    dim3 threshold_threads(128,1,1);
    dim3 threshold_blocks(ri.n_proj_pull/128,1,1);

    find_thresholds<<<threshold_blocks,threshold_threads>>>(d_ecc,d_sup_smooth,d_p_max,d_p_min,d_threshold);
    
    cudaMemcpy(h_threshold,d_threshold,ri.n_proj_pull*sizeof(float),cudaMemcpyDeviceToHost);

    float_debug(h_threshold,ri.n_proj_pull,"/home/john/Desktop/h_threshold.txt");

    // Filter the raw projection data
    float * d_filtered_raw;
    float * h_filtered_raw=(float *)calloc(cg.n_channels*cg.n_rows_raw*ri.n_proj_pull,sizeof(float));
    cudaMalloc(&d_filtered_raw,cg.n_channels*cg.n_rows_raw*ri.n_proj_pull*sizeof(float));
    
    dim3 filter_threads(128,1,1);
    dim3 filter_blocks(ri.n_proj_pull/128,1,1);
    
    filter_projections<<<filter_blocks,filter_threads>>>(d_raw,d_threshold,d_filtered_raw);

    cudaMemcpy(h_filtered_raw,d_filtered_raw,cg.n_channels*cg.n_rows_raw*ri.n_proj_pull*sizeof(float),cudaMemcpyDeviceToHost);
    
    //float_debug(h_filtered_raw,cg.n_channels*cg.n_rows_raw*ri.n_proj_pull,"/home/john/Desktop/h_filtered.txt");
    //float_debug(mr->ctd.raw,cg.n_channels*cg.n_rows_raw*ri.n_proj_pull,"/home/john/Desktop/h_raw.txt");

    // Copy filtered raw data back to raw array
    for (int i=0; i<cg.n_channels*cg.n_rows_raw*ri.n_proj_pull; i++){
	mr->ctd.raw[i]=h_filtered_raw[i];
    }
    
    cudaFree(d_sup);
    cudaFree(d_sup_smooth);
    cudaFree(d_ecc);
    cudaFree(d_p_max);
    cudaFree(d_p_min);
    cudaFree(d_threshold);
    cudaFree(d_raw);
    cudaFree(d_filtered_raw);
    free(h_sup);
    free(h_sup_smooth);
    free(h_threshold);
    free(h_filtered_raw);
    free(h_ecc);
	       
    return 0;
}
