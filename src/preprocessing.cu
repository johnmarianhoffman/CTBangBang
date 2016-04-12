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

    for (int i=0; i<ri.n_proj_pull; i++){
	printf("%.2f\n",h_sup[i]);
    }
    
    cudaFree(d_sup);

    // Smooth the array of maxima
    


    free(h_sup);

    return 0;
}
