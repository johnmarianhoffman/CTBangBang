#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <recon_structs.h>
#include <rebin_filter.cuh>
#include <rebin_filter.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
	{
	    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	    if (abort) exit(code);
	}
}

void copy_sheet(float * sheetptr, int row, struct recon_metadata *mr, struct ct_geom cg);
void load_filter(float * f_array,struct recon_metadata * mr);

void rebin_nffs(struct recon_metadata *mr);
void rebin_pffs(struct recon_metadata *mr);
void rebin_zffs(struct recon_metadata *mr);
void rebin_affs(struct recon_metadata *mr);

int rebin_filter(struct recon_metadata * mr){
    switch (mr->ri.n_ffs){
    case 1:{
	rebin_nffs(mr);
	break;}
    case 2:{
	if (mr->rp.z_ffs==1)
	    rebin_zffs(mr);
	else
	    rebin_pffs(mr);
	break;}
    case 4:{
	rebin_affs(mr);
	break;}
    }
    
    return 0;
}

void rebin_nffs(struct recon_metadata *mr){

    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    struct ct_geom cg=mr->cg;
    struct recon_info ri=mr->ri;
    
    // Allocate the entire output array here and on GPU Note that size
    // of output array is the same for all FFS configurations
    float * d_output;
    float * h_output=(float *)calloc(cg.n_channels_oversampled*cg.n_rows*mr->ri.n_proj_pull/mr->ri.n_ffs,sizeof(float));
    cudaMalloc(&d_output,cg.n_channels_oversampled*cg.n_rows*mr->ri.n_proj_pull/mr->ri.n_ffs*sizeof(float));
    cudaMemset(d_output,0,cg.n_channels_oversampled*cg.n_rows*mr->ri.n_proj_pull/mr->ri.n_ffs*sizeof(float));
    
    // Copy of ct geometry
    cudaMemcpyToSymbol(d_cg,&cg,sizeof(struct ct_geom),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_ri,&ri,sizeof(struct recon_info),0,cudaMemcpyHostToDevice);
    
    // Ready our filter
    float * h_filter=(float*)calloc(2*cg.n_channels_oversampled,sizeof(float));
    float * d_filter;
    cudaMalloc(&d_filter,2*cg.n_channels_oversampled*sizeof(float));
    load_filter(h_filter,mr);
    cudaMemcpy(d_filter,h_filter,2*cg.n_channels_oversampled*sizeof(float),cudaMemcpyHostToDevice);
    
    // Configure textures (see rebin_filter.cuh)
    tex_a.addressMode[0] = cudaAddressModeClamp;
    tex_a.addressMode[1] = cudaAddressModeClamp;
    tex_a.addressMode[2] = cudaAddressModeClamp;
    tex_a.filterMode     = cudaFilterModeLinear;
    tex_a.normalized     = false;

    tex_b.addressMode[0] = cudaAddressModeClamp;
    tex_b.addressMode[1] = cudaAddressModeClamp;
    tex_b.addressMode[2] = cudaAddressModeClamp;
    tex_b.filterMode     = cudaFilterModeLinear;
    tex_b.normalized     = false;

    // Projection data will be thought of as row-sheets (since we row-wise rebin)
    size_t proj_array_size=cg.n_channels*mr->ri.n_proj_pull;
    cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc<float>();

    // Allocate raw data array 1
    float * sheet_1=(float *)calloc(cg.n_channels*mr->ri.n_proj_pull,sizeof(float));
    cudaArray * cu_raw_1;
    cudaMallocArray(&cu_raw_1,&channelDesc,cg.n_channels,mr->ri.n_proj_pull);
    // Allocate raw data array 2
    float * sheet_2=(float *)calloc(cg.n_channels*mr->ri.n_proj_pull,sizeof(float));
    cudaArray * cu_raw_2;
    cudaMallocArray(&cu_raw_2,&channelDesc,cg.n_channels,mr->ri.n_proj_pull);

    // Kernel Dimensions
    dim3 rebin_threads(32,32);
    dim3 rebin_blocks(cg.n_channels_oversampled/rebin_threads.x,mr->ri.n_proj_pull/rebin_threads.y);

    dim3 filter_threads(2*128,1);
    dim3 filter_blocks(mr->ri.n_proj_pull/mr->ri.n_ffs/filter_threads.x,1);

    // Reshape raw data into row sheets
    float * sheets=(float*)calloc(cg.n_channels*cg.n_rows_raw*mr->ri.n_proj_pull,sizeof(float));
    for (int i=0;i<cg.n_rows_raw;i++){
	for (int j=0;j<cg.n_channels;j++){
	    for (int k=0;k<mr->ri.n_proj_pull;k++){
		sheets[j+k*cg.n_channels+i*cg.n_channels*mr->ri.n_proj_pull]=mr->ctd.raw[k*cg.n_channels*cg.n_rows_raw+i*cg.n_channels+j];
	    }
	}
    }
	
    for (int i=0;i<cg.n_rows;i+=2){
	// Copy first set of projections over to GPU	
	cudaMemcpyToArrayAsync(cu_raw_1,0,0,&sheets[i*proj_array_size],proj_array_size*sizeof(float),cudaMemcpyHostToDevice,stream1);
	cudaBindTextureToArray(tex_a,cu_raw_1,channelDesc);

	// Launch Kernel A
	n1_rebin<<<rebin_blocks,rebin_threads,0,stream1>>>(d_output,i);
	filter<<<filter_blocks,filter_threads,0,stream1>>>(d_output,d_filter,i);
	    
	//Begin the second transfer while 1st kernel executes
	cudaMemcpyToArrayAsync(cu_raw_2,0,0,&sheets[(i+1)*proj_array_size],proj_array_size*sizeof(float),cudaMemcpyHostToDevice,stream2);
	cudaBindTextureToArray(tex_b,cu_raw_2,channelDesc);

	n2_rebin<<<rebin_blocks,rebin_threads,0,stream2>>>(d_output,i+1);
	filter<<<filter_blocks,filter_threads,0,stream2>>>(d_output,d_filter,i+1);
    }
	
    cudaFreeArray(cu_raw_1);
    cudaFreeArray(cu_raw_2);

    // Copy final rebinned projections back to host
    cudaMemcpy(h_output,d_output,cg.n_channels_oversampled*cg.n_rows*mr->ri.n_proj_pull/mr->ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost);
    
    //Reshape data into our mr structure
    size_t offset=cg.add_projections;
    for (int i=0;i<cg.n_rows;i++){
	for (int j=0;j<cg.n_channels_oversampled;j++){
	    for (int k=0;k<(mr->ri.n_proj_pull/mr->ri.n_ffs-2*cg.add_projections);k++){
		mr->ctd.rebin[k*cg.n_channels_oversampled*cg.n_rows+i*cg.n_channels_oversampled+j]=h_output[(cg.n_channels_oversampled*mr->ri.n_proj_pull/mr->ri.n_ffs)*i+mr->ri.n_proj_pull/mr->ri.n_ffs*j+(k+offset)];
	    }
	}
    }

    // Check "testing" flag, write rebin to disk if set
    if (mr->flags.testing){
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/rebin.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(mr->ctd.rebin,sizeof(float),cg.n_channels_oversampled*cg.n_rows*(mr->ri.n_proj_pull-2*cg.add_projections_ffs)/mr->ri.n_ffs,outfile);
	fclose(outfile);
    }

    cudaFree(d_output);
    cudaFree(d_filter);
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    free(h_filter);
    free(h_output);
}

void rebin_pffs(struct recon_metadata *mr){

    // Set up some constants on the host 
    struct ct_geom cg=mr->cg;
    struct recon_info ri=mr->ri;

    const double da=cg.src_to_det*cg.r_f*cg.fan_angle_increment/(4.0f*(cg.src_to_det-cg.r_f));

    // Set up some constants and infrastructure on the GPU
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaMemcpyToSymbol(d_cg,&cg,sizeof(struct ct_geom),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_ri,&ri,sizeof(struct recon_info),0,cudaMemcpyHostToDevice);
    
    int proj_per_call=32;
 
    // Need to split arrays by focal spot and reshape into "sheets"
    float * d_raw_1;
    float * d_raw_2;
    cudaMalloc(&d_raw_1,proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float));
    cudaMalloc(&d_raw_2,proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float));

    float * d_fs;
    cudaMalloc(&d_fs,mr->ri.n_proj_pull*cg.n_channels*cg.n_rows_raw*sizeof(float));

    dim3 threads_reshape(32,16,1);
    dim3 blocks_reshape(cg.n_channels/threads_reshape.x,cg.n_rows_raw/threads_reshape.y,proj_per_call/2);
    
    int i=0;
    while (i<mr->ri.n_proj_pull){

	cudaMemcpyAsync(d_raw_1,&mr->ctd.raw[i*cg.n_channels*cg.n_rows_raw],proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float),cudaMemcpyHostToDevice,stream1);
	p_reshape<<<blocks_reshape,threads_reshape,0,stream1>>>(d_raw_1,d_fs,i);
	
	cudaMemcpyAsync(d_raw_2,&mr->ctd.raw[(i+proj_per_call)*cg.n_channels*cg.n_rows_raw],proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float),cudaMemcpyHostToDevice,stream2);
	p_reshape<<<blocks_reshape,threads_reshape,0,stream2>>>(d_raw_2,d_fs,i+proj_per_call);

	i+=2*proj_per_call;
    }

    cudaFree(d_raw_1);
    cudaFree(d_raw_2);

    // Configure textures (see rebin_filter.cuh)
    tex_a.addressMode[0] = cudaAddressModeClamp;
    tex_a.addressMode[1] = cudaAddressModeClamp;
    tex_a.addressMode[2] = cudaAddressModeClamp;
    tex_a.filterMode     = cudaFilterModeLinear;
    tex_a.normalized     = false;

    tex_b.addressMode[0] = cudaAddressModeClamp;
    tex_b.addressMode[1] = cudaAddressModeClamp;
    tex_b.addressMode[2] = cudaAddressModeClamp;
    tex_b.filterMode     = cudaFilterModeLinear;
    tex_b.normalized     = false;

    cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc<float>();

    cudaArray * cu_raw_1;
    cudaArray * cu_raw_2;
    cudaMallocArray(&cu_raw_1,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels);
    cudaMallocArray(&cu_raw_2,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels);
    
    float * d_rebin_t;
    cudaMalloc(&d_rebin_t,cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs*sizeof(float));
    cudaMemset(d_rebin_t,0,cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs*sizeof(float));
    
    dim3 threads_t_rebin(32,32);
    dim3 blocks_t_rebin(cg.n_channels/threads_t_rebin.x,ri.n_proj_pull/ri.n_ffs/threads_t_rebin.y);
    
    for (int i=0;i<cg.n_rows;i++){
	cudaMemcpyToArrayAsync(cu_raw_1,0,0,&d_fs[cg.n_channels*ri.n_proj_pull/ri.n_ffs*i],cg.n_channels*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream1);
	cudaBindTextureToArray(tex_a,cu_raw_1,channelDesc);

	cudaMemcpyToArrayAsync(cu_raw_2,0,0,&d_fs[cg.n_channels*ri.n_proj_pull/ri.n_ffs*i+cg.n_channels*ri.n_proj_pull/ri.n_ffs*cg.n_rows_raw],cg.n_channels*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream2);
	cudaBindTextureToArray(tex_b,cu_raw_2,channelDesc);

	p1_rebin_t<<<blocks_t_rebin,threads_t_rebin,0,stream1>>>(d_rebin_t,da,i);
	p2_rebin_t<<<blocks_t_rebin,threads_t_rebin,0,stream2>>>(d_rebin_t,da,i);
    }
   
    cudaFree(d_fs);
    cudaFreeArray(cu_raw_1);
    cudaFreeArray(cu_raw_2);
    
    if (mr->flags.testing){
	float * h_rebin_t=(float*)calloc(cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs,sizeof(float));
	cudaMemcpy(h_rebin_t,d_rebin_t,cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost);
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/rebin_t.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(h_rebin_t,sizeof(float),cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs,outfile);
	fclose(outfile);
	free(h_rebin_t);
    }

    cudaMallocArray(&cu_raw_1,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels_oversampled);
    cudaMallocArray(&cu_raw_2,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels_oversampled);

    float * d_output;
    cudaMalloc(&d_output,cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs*sizeof(float));

    // Ready our filter
    float * h_filter=(float*)calloc(2*cg.n_channels_oversampled,sizeof(float));
    float * d_filter;
    cudaMalloc(&d_filter,2*cg.n_channels_oversampled*sizeof(float));
    load_filter(h_filter,mr);
    cudaMemcpy(d_filter,h_filter,2*cg.n_channels_oversampled*sizeof(float),cudaMemcpyHostToDevice);
    
    int sheet_size=cg.n_channels_oversampled*ri.n_proj_pull/ri.n_ffs;

    dim3 threads_rebin(32,32);
    dim3 blocks_rebin(cg.n_channels_oversampled/threads_rebin.x,ri.n_proj_pull/ri.n_ffs/threads_rebin.y);

    dim3 threads_filter(2*64,1);
    dim3 blocks_filter(mr->ri.n_proj_pull/mr->ri.n_ffs/threads_filter.x,1);

    for (int i=0;i<cg.n_rows;i+=2){

	cudaMemcpyToArrayAsync(cu_raw_1,0,0,&d_rebin_t[i*sheet_size],sheet_size*sizeof(float),cudaMemcpyDeviceToDevice,stream1);
	cudaBindTextureToArray(tex_a,cu_raw_1,channelDesc);

	cudaMemcpyToArrayAsync(cu_raw_2,0,0,&d_rebin_t[(i+1)*sheet_size],sheet_size*sizeof(float),cudaMemcpyDeviceToDevice,stream2);
	cudaBindTextureToArray(tex_b,cu_raw_2,channelDesc);

	p1_rebin<<<blocks_rebin,threads_rebin,0,stream1>>>(d_output,da,i);
	filter<<<blocks_filter,threads_filter,0,stream1>>>(d_output,d_filter,i);
	    
	p2_rebin<<<blocks_rebin,threads_rebin,0,stream2>>>(d_output,da,i+1);
	filter<<<blocks_filter,threads_filter,0,stream2>>>(d_output,d_filter,i+1);

    }

    float * h_output;
    h_output=(float*)calloc(cg.n_channels_oversampled*cg.n_rows*mr->ri.n_proj_pull/mr->ri.n_ffs,sizeof(float));
    cudaMemcpy(h_output,d_output,cg.n_channels_oversampled*cg.n_rows*mr->ri.n_proj_pull/mr->ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost);

    //Reshape data into our mr structure
    size_t offset=cg.add_projections;
    for (int i=0;i<cg.n_rows;i++){
	for (int j=0;j<cg.n_channels_oversampled;j++){
	    for (int k=0;k<(mr->ri.n_proj_pull/mr->ri.n_ffs-2*cg.add_projections);k++){
		mr->ctd.rebin[k*cg.n_channels_oversampled*cg.n_rows+i*cg.n_channels_oversampled+j]=h_output[(cg.n_channels_oversampled*mr->ri.n_proj_pull/mr->ri.n_ffs)*i+mr->ri.n_proj_pull/mr->ri.n_ffs*j+(k+offset)];
	    }
	}
    }
    
    // Check "testing" flag, write rebin to disk if set
    if (mr->flags.testing){
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/rebin.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(mr->ctd.rebin,sizeof(float),cg.n_channels_oversampled*cg.n_rows*(mr->ri.n_proj_pull-2*cg.add_projections_ffs)/mr->ri.n_ffs,outfile);
	fclose(outfile);
    }

    cudaFree(d_rebin_t);
    cudaFree(d_output);
    cudaFree(d_filter);
    cudaFreeArray(cu_raw_1);
    cudaFreeArray(cu_raw_2);
    
    free(h_output);
    free(h_filter);
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
}
void rebin_zffs(struct recon_metadata *mr){


    // Two GPU parts to this rebin:
    // (1) Split arrays and reshape into sheets
    // (2) Rebin sheets and interleave into final array

    struct ct_geom cg=mr->cg;
    struct recon_info ri=mr->ri;
    struct recon_params rp=mr->rp;

    const double da=0.0;
    const double dr=cg.src_to_det*rp.coll_slicewidth/(4.0*(cg.src_to_det-cg.r_f)*tan(cg.anode_angle));
    
    cudaStream_t stream1,stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Copy of ct geometry
    cudaMemcpyToSymbol(d_cg,&cg,sizeof(struct ct_geom),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_ri,&ri,sizeof(struct recon_info),0,cudaMemcpyHostToDevice);
    
    int proj_per_call=32;

    // Need to split arrays by focal spot and reshape into "sheets"
    float * d_raw_1;
    float * d_raw_2;
    cudaMalloc(&d_raw_1,proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float));
    cudaMalloc(&d_raw_2,proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float));
    
    float * d_fs;
    cudaMalloc(&d_fs,mr->ri.n_proj_pull*cg.n_channels*cg.n_rows_raw*sizeof(float));

    dim3 threads_reshape(32,16,1);
    dim3 blocks_reshape(cg.n_channels/threads_reshape.x,cg.n_rows_raw/threads_reshape.y,proj_per_call/2);
    
    int i=0;
    while (i<mr->ri.n_proj_pull){

	cudaMemcpyAsync(d_raw_1,&mr->ctd.raw[i*cg.n_channels*cg.n_rows_raw],proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float),cudaMemcpyHostToDevice,stream1);
	z_reshape<<<blocks_reshape,threads_reshape,0,stream1>>>(d_raw_1,d_fs,i);
	
	cudaMemcpyAsync(d_raw_2,&mr->ctd.raw[(i+proj_per_call)*cg.n_channels*cg.n_rows_raw],proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float),cudaMemcpyHostToDevice,stream2);
	z_reshape<<<blocks_reshape,threads_reshape,0,stream2>>>(d_raw_2,d_fs,i+proj_per_call);

	i+=2*proj_per_call;
    }

    cudaFree(d_raw_1);
    cudaFree(d_raw_2);

    // Check "testing" flag, write rebin to disk if set
    if (mr->flags.testing){
	float * h_fs_1;
	float * h_fs_2;
	h_fs_1=(float*)calloc((mr->ri.n_proj_pull/mr->ri.n_ffs)*cg.n_channels*cg.n_rows_raw,sizeof(float));
	h_fs_2=(float*)calloc((mr->ri.n_proj_pull/mr->ri.n_ffs)*cg.n_channels*cg.n_rows_raw,sizeof(float));

	cudaMemcpyAsync(h_fs_1, d_fs                                                     ,cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost,stream1);
	cudaMemcpyAsync(h_fs_2,&d_fs[cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs],cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost,stream2);    

	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/reshape_1.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(h_fs_1,sizeof(float),cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,outfile);
	fclose(outfile);

	memset(fullpath,0,4096+255);
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/reshape_2.ct_test");
	outfile=fopen(fullpath,"w");
	fwrite(h_fs_1,sizeof(float),cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,outfile);
	fclose(outfile);

	free(h_fs_1);
	free(h_fs_2);
    }

    
    // Step 1 finished
    // Set up everything for the rebinning
    // Configure textures (see rebin_filter.cuh)
    tex_a.addressMode[0] = cudaAddressModeClamp;
    tex_a.addressMode[1] = cudaAddressModeClamp;
    tex_a.addressMode[2] = cudaAddressModeClamp;
    tex_a.filterMode     = cudaFilterModeLinear;
    tex_a.normalized     = false;

    tex_b.addressMode[0] = cudaAddressModeClamp;
    tex_b.addressMode[1] = cudaAddressModeClamp;
    tex_b.addressMode[2] = cudaAddressModeClamp;
    tex_b.filterMode     = cudaFilterModeLinear;
    tex_b.normalized     = false;

    // Projection data will be thought of as row-sheets (since we row-wise rebin)
    //size_t proj_array_size=cg.n_channels*mr->ri.n_proj_pull;
    cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc<float>();

    cudaArray * cu_raw_1;
    cudaArray * cu_raw_2;
    cudaMallocArray(&cu_raw_1,&channelDesc,mr->ri.n_proj_pull/ri.n_ffs,cg.n_channels);
    cudaMallocArray(&cu_raw_2,&channelDesc,mr->ri.n_proj_pull/ri.n_ffs,cg.n_channels);

    float * d_output;
    cudaMalloc(&d_output,cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs*sizeof(float));

    // Allocate and compute beta lookup tables
    float * d_beta_lookup_1;
    float * d_beta_lookup_2;
    cudaMalloc(&d_beta_lookup_1,cg.n_channels*sizeof(float));
    cudaMalloc(&d_beta_lookup_2,cg.n_channels*sizeof(float));
    beta_lookup<<<1,cg.n_channels>>>(d_beta_lookup_1,-dr,da,0);
    beta_lookup<<<1,cg.n_channels>>>(d_beta_lookup_2, dr,da,0);

    if (mr->flags.testing){
	float * h_b_lookup_1=(float*)calloc(cg.n_channels,sizeof(float));
	float * h_b_lookup_2=(float*)calloc(cg.n_channels,sizeof(float));
	
	cudaMemcpy(h_b_lookup_1,d_beta_lookup_1,cg.n_channels*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b_lookup_2,d_beta_lookup_2,cg.n_channels*sizeof(float),cudaMemcpyDeviceToHost);
	
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/beta_lookup_1.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(h_b_lookup_1,sizeof(float),cg.n_channels,outfile);
	fwrite(h_b_lookup_2,sizeof(float),cg.n_channels,outfile);
	fclose(outfile);

	free(h_b_lookup_1);
	free(h_b_lookup_2);
    }

    // Ready our filter
    float * h_filter=(float*)calloc(2*cg.n_channels_oversampled,sizeof(float));
    float * d_filter;
    cudaMalloc(&d_filter,2*cg.n_channels_oversampled*sizeof(float));
    load_filter(h_filter,mr);
    cudaMemcpy(d_filter,h_filter,2*cg.n_channels_oversampled*sizeof(float),cudaMemcpyHostToDevice);

    dim3 threads_rebin(32,32);
    dim3 blocks_rebin(cg.n_channels_oversampled/threads_rebin.x,ri.n_proj_pull/ri.n_ffs/threads_rebin.y);

    dim3 threads_filter(2*64,1);
    dim3 blocks_filter(mr->ri.n_proj_pull/mr->ri.n_ffs/threads_filter.x,1);
    
    for (int i=0;i<cg.n_rows_raw;i++){
	cudaMemcpyToArrayAsync(cu_raw_1,0,0,&d_fs[cg.n_channels*ri.n_proj_pull/ri.n_ffs*i],cg.n_channels*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream1);
	cudaBindTextureToArray(tex_a,cu_raw_1,channelDesc);

	cudaMemcpyToArrayAsync(cu_raw_2,0,0,&d_fs[cg.n_channels*ri.n_proj_pull/ri.n_ffs*i+cg.n_channels*ri.n_proj_pull/ri.n_ffs*cg.n_rows_raw],cg.n_channels*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream2);
	cudaBindTextureToArray(tex_b,cu_raw_2,channelDesc);

	z1_rebin<<<blocks_rebin,threads_rebin,0,stream1>>>(d_output,d_beta_lookup_1,dr,i);
	filter<<<blocks_filter,threads_filter,0,stream1>>>(d_output,d_filter,2*i);
	
	z2_rebin<<<blocks_rebin,threads_rebin,0,stream2>>>(d_output,d_beta_lookup_2,dr,i);
	filter<<<blocks_filter,threads_filter,0,stream2>>>(d_output,d_filter,2*i+1);
    }

    float * h_output;
    h_output=(float*)calloc(cg.n_channels_oversampled*cg.n_rows*mr->ri.n_proj_pull/mr->ri.n_ffs,sizeof(float));
    cudaMemcpy(h_output,d_output,cg.n_channels_oversampled*cg.n_rows*mr->ri.n_proj_pull/mr->ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost);

    //Reshape data into our mr structure
    size_t offset=cg.add_projections;
    for (int i=0;i<cg.n_rows;i++){
	for (int j=0;j<cg.n_channels_oversampled;j++){
	    for (int k=0;k<(mr->ri.n_proj_pull/mr->ri.n_ffs-2*cg.add_projections);k++){
		mr->ctd.rebin[k*cg.n_channels_oversampled*cg.n_rows+i*cg.n_channels_oversampled+j]=h_output[(cg.n_channels_oversampled*mr->ri.n_proj_pull/mr->ri.n_ffs)*i+mr->ri.n_proj_pull/mr->ri.n_ffs*j+(k+offset)];
	    }
	}
    }
    
    // Check "testing" flag, write rebin to disk if set
    if (mr->flags.testing){
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/rebin.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(mr->ctd.rebin,sizeof(float),cg.n_channels_oversampled*cg.n_rows*(mr->ri.n_proj_pull-2*cg.add_projections_ffs)/mr->ri.n_ffs,outfile);
	fclose(outfile);
    }

    cudaFree(d_fs);
    cudaFree(d_output);
    cudaFree(d_beta_lookup_1);
    cudaFree(d_beta_lookup_2);
    cudaFree(d_filter);
    
    cudaFreeArray(cu_raw_1);
    cudaFreeArray(cu_raw_2);
    
    cudaStreamDestroy(stream1); 
    cudaStreamDestroy(stream2);
    
}
void rebin_affs(struct recon_metadata *mr){

    // Set up some constants on the host 
    struct ct_geom cg=mr->cg;
    struct recon_info ri=mr->ri;
    struct recon_params rp=mr->rp;

    const float da=cg.src_to_det*cg.r_f*cg.fan_angle_increment/(4.0f*(cg.src_to_det-cg.r_f));
    const float dr=cg.src_to_det*rp.coll_slicewidth/(4.0*(cg.src_to_det-cg.r_f)*tan(cg.anode_angle));
    
    // Set up some constants and infrastructure on the GPU
    cudaStream_t stream1,stream2,stream3,stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    cudaMemcpyToSymbol(d_cg,&cg,sizeof(struct ct_geom),0,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_ri,&ri,sizeof(struct recon_info),0,cudaMemcpyHostToDevice);

    // Configure textures (see rebin_filter.cuh)
    tex_a.addressMode[0] = cudaAddressModeClamp;
    tex_a.addressMode[1] = cudaAddressModeClamp;
    tex_a.addressMode[2] = cudaAddressModeClamp;
    tex_a.filterMode     = cudaFilterModeLinear;
    tex_a.normalized     = false;

    tex_b.addressMode[0] = cudaAddressModeClamp;
    tex_b.addressMode[1] = cudaAddressModeClamp;
    tex_b.addressMode[2] = cudaAddressModeClamp;
    tex_b.filterMode     = cudaFilterModeLinear;
    tex_b.normalized     = false;

    cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc<float>();

    // Need to split arrays by focal spot and reshape into "sheets"
    int proj_per_call=32;

    float * d_raw_1;
    float * d_raw_2;
    float * d_raw_3;
    float * d_raw_4;
    cudaMalloc(&d_raw_1,proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float));
    cudaMalloc(&d_raw_2,proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float));
    cudaMalloc(&d_raw_3,proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float));
    cudaMalloc(&d_raw_4,proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float));
    
    float * d_fs;
    cudaMalloc(&d_fs,mr->ri.n_proj_pull*cg.n_channels*cg.n_rows_raw*sizeof(float));

    dim3 threads_reshape(32,16,1);
    dim3 blocks_reshape(cg.n_channels/threads_reshape.x,cg.n_rows_raw/threads_reshape.y,proj_per_call/4);
    
    int i=0;
    while (i<mr->ri.n_proj_pull){

	cudaMemcpyAsync(d_raw_1,&mr->ctd.raw[i*cg.n_channels*cg.n_rows_raw],proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float),cudaMemcpyHostToDevice,stream1);
	a_reshape<<<blocks_reshape,threads_reshape,0,stream1>>>(d_raw_1,d_fs,i);
	
	cudaMemcpyAsync(d_raw_2,&mr->ctd.raw[(i+proj_per_call)*cg.n_channels*cg.n_rows_raw],proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float),cudaMemcpyHostToDevice,stream2);
	a_reshape<<<blocks_reshape,threads_reshape,0,stream2>>>(d_raw_2,d_fs,i+proj_per_call);

	cudaMemcpyAsync(d_raw_3,&mr->ctd.raw[(i+2*proj_per_call)*cg.n_channels*cg.n_rows_raw],proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float),cudaMemcpyHostToDevice,stream3);
	a_reshape<<<blocks_reshape,threads_reshape,0,stream3>>>(d_raw_3,d_fs,i+2*proj_per_call);

	cudaMemcpyAsync(d_raw_4,&mr->ctd.raw[(i+3*proj_per_call)*cg.n_channels*cg.n_rows_raw],proj_per_call*cg.n_channels*cg.n_rows_raw*sizeof(float),cudaMemcpyHostToDevice,stream4);
	a_reshape<<<blocks_reshape,threads_reshape,0,stream4>>>(d_raw_4,d_fs,i+3*proj_per_call);

	i+=4*proj_per_call;
    }

    // Check "testing" flag, write rebin to disk if set
    if (mr->flags.testing){
	float * h_fs_1;
	float * h_fs_2;
	float * h_fs_3;
	float * h_fs_4;
	h_fs_1=(float*)calloc((mr->ri.n_proj_pull/mr->ri.n_ffs)*cg.n_channels*cg.n_rows_raw,sizeof(float));
	h_fs_2=(float*)calloc((mr->ri.n_proj_pull/mr->ri.n_ffs)*cg.n_channels*cg.n_rows_raw,sizeof(float));
	h_fs_3=(float*)calloc((mr->ri.n_proj_pull/mr->ri.n_ffs)*cg.n_channels*cg.n_rows_raw,sizeof(float));
	h_fs_4=(float*)calloc((mr->ri.n_proj_pull/mr->ri.n_ffs)*cg.n_channels*cg.n_rows_raw,sizeof(float));

	cudaMemcpyAsync(h_fs_1, d_fs                                                     ,cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost,stream1);
	cudaMemcpyAsync(h_fs_2,&d_fs[  cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs],cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost,stream2);    
	cudaMemcpyAsync(h_fs_3,&d_fs[2*cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs],cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost,stream3);
	cudaMemcpyAsync(h_fs_4,&d_fs[3*cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs],cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost,stream4);    

	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/reshape_1.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(h_fs_1,sizeof(float),cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,outfile);
	fclose(outfile);

	memset(fullpath,0,4096+255);
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/reshape_2.ct_test");
	outfile=fopen(fullpath,"w");
	fwrite(h_fs_1,sizeof(float),cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,outfile);
	fclose(outfile);

	memset(fullpath,0,4096+255);
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/reshape_3.ct_test");
	outfile=fopen(fullpath,"w");
	fwrite(h_fs_3,sizeof(float),cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,outfile);
	fclose(outfile);

	memset(fullpath,0,4096+255);
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/reshape_4.ct_test");
	outfile=fopen(fullpath,"w");
	fwrite(h_fs_4,sizeof(float),cg.n_channels*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,outfile);
	fclose(outfile);

	free(h_fs_1);
	free(h_fs_2);
	free(h_fs_3);
	free(h_fs_4);
    }
    
    cudaFree(d_raw_1);
    cudaFree(d_raw_2);
    cudaFree(d_raw_3);
    cudaFree(d_raw_4);

    float * d_rebin_t1;
    float * d_rebin_t2;
    cudaMalloc(&d_rebin_t1,ri.n_proj_pull/ri.n_ffs*cg.n_channels_oversampled*cg.n_rows_raw*sizeof(float));
    cudaMalloc(&d_rebin_t2,ri.n_proj_pull/ri.n_ffs*cg.n_channels_oversampled*cg.n_rows_raw*sizeof(float));

    cudaArray * cu_raw_1;
    cudaArray * cu_raw_2;
    cudaArray * cu_raw_3;
    cudaArray * cu_raw_4;    
    cudaMallocArray(&cu_raw_1,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels);
    cudaMallocArray(&cu_raw_2,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels);
    cudaMallocArray(&cu_raw_3,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels);
    cudaMallocArray(&cu_raw_4,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels);

    dim3 threads_t_rebin(32,32);
    dim3 blocks_t_rebin(cg.n_channels/threads_t_rebin.x,ri.n_proj_pull/ri.n_ffs/threads_t_rebin.y);
    
    for (int i=0;i<cg.n_rows_raw;i++){
	cudaMemcpyToArrayAsync(cu_raw_1,0,0,&d_fs[cg.n_channels*ri.n_proj_pull/ri.n_ffs*i],cg.n_channels*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream1);
	cudaBindTextureToArray(tex_a,cu_raw_1,channelDesc);
	a1_rebin_t<<<blocks_t_rebin,threads_t_rebin,0,stream1>>>(d_rebin_t1,da,dr,i);
	
	cudaMemcpyToArrayAsync(cu_raw_2,0,0,&d_fs[cg.n_channels*ri.n_proj_pull/ri.n_ffs*i+cg.n_channels*ri.n_proj_pull/ri.n_ffs*cg.n_rows_raw],cg.n_channels*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream2);
	cudaBindTextureToArray(tex_b,cu_raw_2,channelDesc);
	a2_rebin_t<<<blocks_t_rebin,threads_t_rebin,0,stream2>>>(d_rebin_t1,da,dr,i);
	
	cudaMemcpyToArrayAsync(cu_raw_3,0,0,&d_fs[cg.n_channels*ri.n_proj_pull/ri.n_ffs*i+2*cg.n_channels*ri.n_proj_pull/ri.n_ffs*cg.n_rows_raw],cg.n_channels*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream3);
	cudaBindTextureToArray(tex_c,cu_raw_3,channelDesc);
	a3_rebin_t<<<blocks_t_rebin,threads_t_rebin,0,stream3>>>(d_rebin_t2,da,dr,i);
	
	cudaMemcpyToArrayAsync(cu_raw_4,0,0,&d_fs[cg.n_channels*ri.n_proj_pull/ri.n_ffs*i+3*cg.n_channels*ri.n_proj_pull/ri.n_ffs*cg.n_rows_raw],cg.n_channels*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream4);
	cudaBindTextureToArray(tex_d,cu_raw_4,channelDesc);
	a4_rebin_t<<<blocks_t_rebin,threads_t_rebin,0,stream4>>>(d_rebin_t2,da,dr,i);
	
    }

    if (mr->flags.testing){
	float * h_rebin_t1;
	float * h_rebin_t2;
	h_rebin_t1=(float*)calloc(cg.n_channels_oversampled*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,sizeof(float));	
	h_rebin_t2=(float*)calloc(cg.n_channels_oversampled*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,sizeof(float));
	
	cudaMemcpy(h_rebin_t1,d_rebin_t1,cg.n_channels_oversampled*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rebin_t2,d_rebin_t2,cg.n_channels_oversampled*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost);

	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/rebin_t1.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(h_rebin_t1,sizeof(float),cg.n_channels_oversampled*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,outfile);
	fclose(outfile);
	
	memset(fullpath,0,4096+255);
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/rebin_t2.ct_test");
	outfile=fopen(fullpath,"w");
	fwrite(h_rebin_t2,sizeof(float),cg.n_channels_oversampled*cg.n_rows_raw*ri.n_proj_pull/ri.n_ffs,outfile);
	fclose(outfile);

	free(h_rebin_t1);
	free(h_rebin_t2);	
    }

    
    cudaFree(d_fs);
    cudaFreeArray(cu_raw_1);
    cudaFreeArray(cu_raw_2);
    cudaFreeArray(cu_raw_3);
    cudaFreeArray(cu_raw_4);

    float * d_output;
    gpuErrchk(cudaMalloc(&d_output,cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs*sizeof(float)));

    gpuErrchk(cudaMallocArray(&cu_raw_1,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels_oversampled));
    gpuErrchk(cudaMallocArray(&cu_raw_2,&channelDesc,ri.n_proj_pull/ri.n_ffs,cg.n_channels_oversampled));

    // Allocate and compute beta lookup tables
    float * d_beta_lookup_1;
    float * d_beta_lookup_2;
    gpuErrchk(cudaMalloc(&d_beta_lookup_1,cg.n_channels_oversampled*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_beta_lookup_2,cg.n_channels_oversampled*sizeof(float)));
    beta_lookup<<<2,cg.n_channels>>>(d_beta_lookup_1,-dr,0,1);
    beta_lookup<<<2,cg.n_channels>>>(d_beta_lookup_2, dr,0,1);

    if (mr->flags.testing){
	float * h_b_lookup_1=(float*)calloc(cg.n_channels_oversampled,sizeof(float));
	float * h_b_lookup_2=(float*)calloc(cg.n_channels_oversampled,sizeof(float));
	
	cudaMemcpy(h_b_lookup_1,d_beta_lookup_1,cg.n_channels_oversampled*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b_lookup_2,d_beta_lookup_2,cg.n_channels_oversampled*sizeof(float),cudaMemcpyDeviceToHost);
	
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/beta_lookup_1.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(h_b_lookup_1,sizeof(float),cg.n_channels_oversampled,outfile);
	fclose(outfile);

	memset(fullpath,0,4096+255);
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/beta_lookup_2.ct_test");
	fopen(fullpath,"w");
	fwrite(h_b_lookup_2,sizeof(float),cg.n_channels_oversampled,outfile);
	fclose(outfile);
	
	free(h_b_lookup_1);
	free(h_b_lookup_2);
    }

    // Ready our filter
    float * h_filter=(float*)calloc(2*cg.n_channels_oversampled,sizeof(float));
    float * d_filter;
    cudaMalloc(&d_filter,2*cg.n_channels_oversampled*sizeof(float));
    load_filter(h_filter,mr);
    cudaMemcpy(d_filter,h_filter,2*cg.n_channels_oversampled*sizeof(float),cudaMemcpyHostToDevice);

    if (mr->flags.testing){
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/filter.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(h_filter,sizeof(float),2*cg.n_channels_oversampled,outfile);
	fclose(outfile);
    }
    
    dim3 threads_rebin(32,32);
    dim3 blocks_rebin(cg.n_channels_oversampled/threads_rebin.x,ri.n_proj_pull/ri.n_ffs/threads_rebin.y);

    dim3 threads_filter(2*64,1);
    dim3 blocks_filter(mr->ri.n_proj_pull/mr->ri.n_ffs/threads_filter.x,1);
    
    for (int i=0;i<cg.n_rows_raw;i++){
	cudaMemcpyToArrayAsync(cu_raw_1,0,0,&d_rebin_t1[cg.n_channels_oversampled*ri.n_proj_pull/ri.n_ffs*i],cg.n_channels_oversampled*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream1);
	cudaBindTextureToArray(tex_a,cu_raw_1,channelDesc);

	cudaMemcpyToArrayAsync(cu_raw_2,0,0,&d_rebin_t2[cg.n_channels_oversampled*ri.n_proj_pull/ri.n_ffs*i],cg.n_channels_oversampled*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToDevice,stream2);
	cudaBindTextureToArray(tex_b,cu_raw_2,channelDesc);

	a1_rebin_b<<<blocks_rebin,threads_rebin,0,stream1>>>(d_output,d_beta_lookup_1,dr,i);
	filter<<<blocks_filter,threads_filter,0,stream1>>>(d_output,d_filter,2*i);
	
	a2_rebin_b<<<blocks_rebin,threads_rebin,0,stream2>>>(d_output,d_beta_lookup_2,dr,i);
	filter<<<blocks_filter,threads_filter,0,stream2>>>(d_output,d_filter,2*i+1);
    }

    float * h_output;
    h_output=(float*)calloc(cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs,sizeof(float));
    cudaMemcpy(h_output,d_output,cg.n_channels_oversampled*cg.n_rows*ri.n_proj_pull/ri.n_ffs*sizeof(float),cudaMemcpyDeviceToHost);
    
    //Reshape data into our mr structure
    size_t offset=cg.add_projections;
    for (int i=0;i<cg.n_rows;i++){
	for (int j=0;j<cg.n_channels_oversampled;j++){
	    for (int k=0;k<(ri.n_proj_pull/ri.n_ffs-2*cg.add_projections);k++){
		mr->ctd.rebin[k*cg.n_channels_oversampled*cg.n_rows+i*cg.n_channels_oversampled+j]=h_output[(cg.n_channels_oversampled*ri.n_proj_pull/ri.n_ffs)*i+ri.n_proj_pull/ri.n_ffs*j+(k+offset)];
	    }
	}
    }
    
    // Check "testing" flag, write rebin to disk if set
    if (mr->flags.testing){
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/rebin.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(mr->ctd.rebin,sizeof(float),cg.n_channels_oversampled*cg.n_rows*(ri.n_proj_pull-2*cg.add_projections_ffs)/ri.n_ffs,outfile);
	fclose(outfile);
    }


    free(h_output);

    cudaFree(d_rebin_t1);
    cudaFree(d_rebin_t2);
    
    cudaFreeArray(cu_raw_1);
    cudaFreeArray(cu_raw_2);
    
    cudaFree(d_output);
    cudaFree(d_filter);
    cudaFree(d_beta_lookup_1);
    cudaFree(d_beta_lookup_2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
}



void copy_sheet(float * sheetptr, int row,struct recon_metadata * mr,struct ct_geom cg){
    for (int j=0;j<cg.n_channels;j++){
	for (int k=0;k<mr->ri.n_proj_pull;k++){
	    sheetptr[j+k*cg.n_channels]=mr->ctd.raw[k*cg.n_channels*cg.n_rows_raw+row*cg.n_channels+j];
	}
    }
}

void load_filter(float * f_array,struct recon_metadata * mr){
    struct ct_geom cg=mr->cg;
    struct recon_params rp=mr->rp;

    char fullpath[4096+255]={0};

    FILE * filter_file;
    switch (rp.recon_kernel){
    case -100:{
	sprintf(fullpath,"%s/resources/filters/f_%i_ramp.txt",mr->install_dir,cg.n_channels);
	break;}
    case -1:{
	sprintf(fullpath,"%s/resources/filters/f_%i_exp.txt",mr->install_dir,cg.n_channels);
	break;}
    default:{
	sprintf(fullpath,"%s/resources/filters/f_%i_b%i.txt",mr->install_dir,cg.n_channels,rp.recon_kernel);
	break;}
    }

    filter_file=fopen(fullpath,"r");

    fread(f_array,sizeof(float),2*cg.n_channels_oversampled,filter_file);
    fclose(filter_file);
}
