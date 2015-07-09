#define pi 3.1415926535897f

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>

#include <interp.h>
#include <recon_structs.h>
#include <rebin_filter_cpu.h>

void copy_sheet(float * sheetptr, int row, struct recon_metadata *mr, struct ct_geom cg);
void load_filter(float * f_array,struct recon_metadata * mr);

void filter_cpu(float * row, float * filter, int N);

void rebin_nffs_cpu(struct recon_metadata *mr);
void rebin_pffs_cpu(struct recon_metadata *mr);
void rebin_zffs_cpu(struct recon_metadata *mr);
void rebin_affs_cpu(struct recon_metadata *mr);

int rebin_filter_cpu(struct recon_metadata * mr){

    switch (mr->ri.n_ffs){
    case 1:{
	rebin_nffs_cpu(mr);
	break;}
    case 2:{
	if (mr->rp.z_ffs==1)
	    rebin_zffs_cpu(mr);
	else
	    rebin_pffs_cpu(mr);
	break;}
    case 4:{
	rebin_affs_cpu(mr);
	break;}
    }
    
    return 0;
}

void rebin_nffs_cpu(struct recon_metadata *mr){
    const struct ct_geom cg=mr->cg;

    float * h_output=(float*)calloc(cg.n_channels_oversampled*cg.n_rows*mr->ri.n_proj_pull/mr->ri.n_ffs,sizeof(float));

    // Main loop
    int n_proj=mr->ri.n_proj_pull/mr->ri.n_ffs;
    struct array_dims d;
    d.idx1=cg.n_channels;
    d.idx2=cg.n_rows;
    d.idx3=n_proj;
    
    for (int channel=0;channel<cg.n_channels_oversampled;channel++){
	const float beta=asin(((float)channel-2*cg.central_channel)*(cg.fan_angle_increment/2));
	float beta_idx=beta/cg.fan_angle_increment+cg.central_channel;
	for (int proj=0;proj<n_proj;proj++){
	    float alpha_idx=(float)proj-beta*cg.n_proj_turn/(2.0f*pi);
	    for (int row=0;row<cg.n_rows;row++){
		int out_idx=cg.n_channels_oversampled*cg.n_rows*proj+cg.n_channels_oversampled*row+channel;
		h_output[out_idx]=interp3(mr->ctd.raw,d,beta_idx,row,alpha_idx);
	    }
	}
    }

    //Copy data into our mr structure, skipping initial truncated projections
    size_t offset=cg.add_projections;
    for (int i=0;i<cg.n_channels_oversampled;i++){
	for (int j=0;j<cg.n_rows;j++){
	    for (int k=0;k<(mr->ri.n_proj_pull/mr->ri.n_ffs-2*cg.add_projections);k++){
		mr->ctd.rebin[k*cg.n_channels_oversampled*cg.n_rows+j*cg.n_channels_oversampled+i]=h_output[(k+offset)*cg.n_channels_oversampled*cg.n_rows+j*cg.n_channels_oversampled+i];
	    }
	}
    }

    
    printf("Filtering...\n");
    
    // Load and run filter
    float * h_filter=(float*)calloc(2*cg.n_channels_oversampled,sizeof(float));
    load_filter(h_filter,mr);

    for (int i=0;i<(n_proj-2*cg.add_projections);i++){
	for (int j=0;j<cg.n_rows;j++){
	    int row_start_idx=i*cg.n_channels_oversampled*cg.n_rows+cg.n_channels_oversampled*j;
	    filter_cpu(&mr->ctd.rebin[row_start_idx],h_filter,cg.n_channels_oversampled);
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

    free(h_output);
    free(h_filter);
}

void rebin_pffs_cpu(struct recon_metadata *mr){

}

void rebin_zffs_cpu(struct recon_metadata *mr){

}

void rebin_affs_cpu(struct recon_metadata *mr){

}

void filter_cpu(float * row, float * filter, int N){
    // N is the number of elements in a row

    // Calculate padding
    int M=2*pow(2.0f,ceil(log2((float)N)));

    // Create two new padded/manipulated vectors from our inputs into fftw complex arrays
    fftw_complex * R = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*M);
    memset(R,0,M*sizeof(fftw_complex));
    for (int i=0;i<N;i++){
	R[i]=row[i];
    }

    fftw_complex * F=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*M);
    memset(F,0,M*sizeof(fftw_complex));
    
    for (int i=0;i<N;i++){
	F[i]=filter[(int)floor((2.0f*N-1.0f)/2.0f)+1+i];
    }
    for (int i=(M-N+1);i<M;i++){
	F[i]=filter[i-(M-N+1)+1];
    }

    // Allocate complex output vectors for row and filter FFTs
    fftw_complex * R_fourier=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*M);
    fftw_complex * F_fourier=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*M);

    // Create plans and execute FFTs
    fftw_plan p_R,p_F;
    p_R=fftw_plan_dft_1d(M,R,R_fourier,FFTW_FORWARD,FFTW_ESTIMATE);
    p_F=fftw_plan_dft_1d(M,F,F_fourier,FFTW_FORWARD,FFTW_ESTIMATE);
    fftw_execute(p_R);
    fftw_execute(p_F);

    //Multiply row and filter into output array
    fftw_complex * O_fourier=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*M);
    for (int i=0;i<M;i++){
	O_fourier[i]=R_fourier[i]*F_fourier[i];
    }

    //Prep final output array and plan, then execute
    fftw_complex * O=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*M);
    fftw_plan p_O;
    p_O=fftw_plan_dft_1d(M,O_fourier,O,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p_O);

    //Copy real portion of final result into source row
    for (int i=0;i<N;i++){
	row[i]=(1.0f/(float)M)*(float)creal(O[i]);
    }

    // Clean up
    fftw_destroy_plan(p_R);
    fftw_destroy_plan(p_F);
    fftw_destroy_plan(p_O);    

    fftw_free(F);
    fftw_free(F_fourier);
    fftw_free(R);
    fftw_free(R_fourier);
    fftw_free(O);
    fftw_free(O_fourier);    
}
