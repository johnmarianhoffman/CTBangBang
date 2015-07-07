#define pi 3.1415926535897f

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <recon_structs.h>
#include <rebin_filter_cpu.h>

void copy_sheet(float * sheetptr, int row, struct recon_metadata *mr, struct ct_geom cg);
void load_filter(float * f_array,struct recon_metadata * mr);
//float interp1(float * array, float idx_1);
//float interp2(float * array, float idx_1, float idx_2);
float interp_nffs(float * array, int array_dims[3],float channel, float proj, int row);

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
    int array_dims[3]={cg.n_channels,cg.n_rows,n_proj};
    
    for (int channel=0;channel<cg.n_channels_oversampled;channel++){
	const float beta=asin(((float)channel-2*cg.central_channel)*(cg.fan_angle_increment/2));
	float beta_idx=beta/cg.fan_angle_increment+cg.central_channel;
	for (int proj=0;proj<n_proj;proj++){
	    float alpha_idx=(float)proj-beta*cg.n_proj_turn/(2.0f*pi);
	    for (int row=0;row<cg.n_rows;row++){
		int out_idx=cg.n_channels_oversampled*cg.n_rows*proj+cg.n_channels_oversampled*row+channel;
		h_output[out_idx]=interp_nffs(mr->ctd.raw,array_dims,beta_idx,alpha_idx,row);
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

    exit(0);
}

void rebin_pffs_cpu(struct recon_metadata *mr){

}

void rebin_zffs_cpu(struct recon_metadata *mr){

}

void rebin_affs_cpu(struct recon_metadata *mr){

}

//float interp1(float * array, float idx_1){
//    
//}
//
//float interp2(float * array, float idx_1, float idx_2){
//
//}

float interp_nffs(float * array,int array_dims[3], float channel, float proj, int row){
    // Array dims is organized as follows: [n_channels n_rows n_proj]

    //Configure edge clamping
    if (channel>(array_dims[0]-2))
	channel=array_dims[0]-2;
    if (channel<0)
	channel=0;
 
    if (proj>(array_dims[2]-2))
	proj=array_dims[2]-2;
    if (proj<0)
	proj=0;
    
    float u=channel-floor(channel);
    float v=proj-floor(proj);

    int a_idx_1=array_dims[0]*array_dims[1]*   (int)floor(proj)   + array_dims[0] * row +   (int)floor(channel);
    int a_idx_2=array_dims[0]*array_dims[1]*   (int)floor(proj)   + array_dims[0] * row +  (int)floor(channel) + 1;
    int a_idx_3=array_dims[0]*array_dims[1]*  (int)floor(proj)+1  + array_dims[0] * row +   (int)floor(channel);    
    int a_idx_4=array_dims[0]*array_dims[1]*  (int)floor(proj)+1  + array_dims[0] * row +  (int)floor(channel) + 1;

    if (a_idx_2-a_idx_1!=1||a_idx_4-a_idx_3!=1){
	printf("%d, %d",a_idx_2-a_idx_1,a_idx_4-a_idx_3);
	exit(0);
    }

    //return v;
    
    return (array[a_idx_1] * (1.0f-u) * (1.0f-v) 
    	  + array[a_idx_2] *     u    * (1.0f-v)
    	  + array[a_idx_3] * (1.0f-u) *    v	 
    	  + array[a_idx_4] *     u    *    v     );

}
