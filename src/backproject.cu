#include <stdlib.h>
#include <stdio.h>
#include <recon_structs.h>
#include <backproject.cuh>
#include <backproject.h>

#define Bx 32
#define By 32

int backproject(struct recon_metadata * mr){
    
    struct ct_geom cg=mr->cg;

    float tube_start=mr->tube_angles[mr->ri.idx_pull_start+cg.add_projections_ffs]*pi/180;
    int n_half_turns=(mr->ri.n_proj_pull/mr->ri.n_ffs-2*cg.add_projections)/(cg.n_proj_turn/2);
    
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Allocate the final volume array
    float * d_output;
    cudaMalloc(&d_output,mr->rp.nx*mr->rp.ny*mr->ri.n_slices_recon*sizeof(float));
    cudaMemset(d_output,0,mr->rp.nx*mr->rp.ny*mr->ri.n_slices_recon*sizeof(float));
    
    // Copy reference structures to device
    cudaMemcpyToSymbol(d_cg,&mr->cg,sizeof(struct ct_geom),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_rp,&mr->rp,sizeof(struct recon_params),0,cudaMemcpyHostToDevice);
    
    // Configure textures (see backproject.cuh)
    tex_a.addressMode[0] = cudaAddressModeBorder;
    tex_a.addressMode[1] = cudaAddressModeBorder;
    tex_a.addressMode[2] = cudaAddressModeBorder;
    tex_a.filterMode     = cudaFilterModeLinear;
    tex_a.normalized     = false;

    tex_b.addressMode[0] = cudaAddressModeBorder;
    tex_b.addressMode[1] = cudaAddressModeBorder;
    tex_b.addressMode[2] = cudaAddressModeBorder;
    tex_b.filterMode     = cudaFilterModeLinear;
    tex_b.normalized     = false;

    cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc<float>();

    cudaArray * cu_proj_1;
    cudaArray * cu_proj_2;
    cudaMallocArray(&cu_proj_1,&channelDesc,cg.n_channels_oversampled,I*cg.n_rows*n_half_turns);
    cudaMallocArray(&cu_proj_2,&channelDesc,cg.n_channels_oversampled,I*cg.n_rows*n_half_turns);

    dim3 threads(Bx,By);
    dim3 blocks(mr->rp.nx/Bx,mr->rp.ny/By,mr->ri.n_slices_recon/K);

    FILE * outfile;

    for (int i=0;i<cg.n_proj_turn/2;i+=I*2){
	for (int k=0;k<n_half_turns;k++){
	    cudaMemcpyToArrayAsync(cu_proj_1,0,k*I*cg.n_rows,&mr->ctd.rebin[(i+k*cg.n_proj_turn/2)*cg.n_rows*cg.n_channels_oversampled],I*cg.n_rows*cg.n_channels_oversampled*sizeof(float),cudaMemcpyHostToDevice,stream1);
	}
	cudaBindTextureToArray(tex_a,cu_proj_1,channelDesc);

	for (int k=0;k<n_half_turns;k++){
	    cudaMemcpyToArrayAsync(cu_proj_2,0,k*I*cg.n_rows,&mr->ctd.rebin[(i+I+k*cg.n_proj_turn/2)*cg.n_rows*cg.n_channels_oversampled],I*cg.n_rows*cg.n_channels_oversampled*sizeof(float),cudaMemcpyHostToDevice,stream2);
	}
	cudaBindTextureToArray(tex_b,cu_proj_2,channelDesc);
	
	// Kernel call 1
	bp_a<<<blocks,threads,0,stream1>>>(d_output,i,tube_start+pi/4.0f,n_half_turns);

	// Kernel call 2
	bp_b<<<blocks,threads,0,stream2>>>(d_output,i+I,tube_start+pi/4.0f,n_half_turns);
	
    }

    long block_offset=(mr->ri.cb.block_idx-1)*mr->rp.nx*mr->rp.ny*mr->ri.n_slices_block;
    cudaMemcpy(&mr->ctd.image[block_offset],d_output,mr->rp.nx*mr->rp.ny*mr->ri.n_slices_block*sizeof(float),cudaMemcpyDeviceToHost);

    outfile=fopen("/home/john/Desktop/image_data.txt","w");
    fwrite(mr->ctd.image,sizeof(float),mr->rp.nx*mr->rp.ny*mr->ri.n_slices_recon,outfile);
    fclose(outfile);

    cudaFree(d_output);
    cudaFreeArray(cu_proj_1);
    cudaFreeArray(cu_proj_2);
  
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
