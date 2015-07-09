#include <stdlib.h>
#include <stdio.h>
#include <reconstructs.h>
#include <backproject_cpu.h>

#define Bx 32
#define By 32
#define K 4
#define I 32

int backproject_cpu(struct recon_metadata * mr){

    struct ct_geom cg=mr->cg;
    struct recon_params rp=mr->rp;
    
    float tube_start=mr->tube_angles[mr->ri.idx_pull_start+cg.add_projections_ffs]*pi/180;
    int n_half_turns=(mr->ri.n_proj_pull/mr->ri.n_ffs-2*cg.add_projections)/(cg.n_proj_turn/2);

    // Allocate the final output volume
    float * h_output;
    h_output=(float *)calloc(mr->rp.nx*mr->rp.ny*mr->ri.n_slices_recon,sizeof(float));

    // Allocate the array to hold the projections currently being processed
    float * proj;
    proj=(float *)malloc(cg.n_channels_oversampled*cg.n_rows*I*n_half_turns*sizeof(float));

    for (int i=0;i<cg.n_proj_turn/2;i+=I){
	for (int k=0;k<n_half_turns;k++){
	    
	    
	    
	}	
    }
    
}
