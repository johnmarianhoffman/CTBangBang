#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

#include <setup.h>
#include <read_raw_file.h>

#define pi 3.1415926535897f
#define BLOCK_SLICES 32

int array_search(float key,double * array,int numel_array,int search_type);

struct recon_params configure_recon_params(char * filename){
    struct recon_params prms;
    memset(&prms, 0,sizeof(prms));
    
    char * prm_buffer;
    char *token;

    FILE * prm_file;
    prm_file=fopen(filename,"r");
    if (prm_file==NULL){
	perror("Parameter file not found.");
	exit(1);
    }

    fseek(prm_file, 0, SEEK_END);
    size_t prm_size = ftell(prm_file);
    rewind(prm_file);
    prm_buffer = (char*)malloc(prm_size + 1);
    prm_buffer[prm_size] = '\0';
    fread(prm_buffer, sizeof(char), prm_size, prm_file);
    fclose(prm_file);

    token=strtok(prm_buffer,"\t\n");

    //Parse parameter file
    while (token!=NULL){
	if (strcmp(token,"RawDataDir:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%s",prms.raw_data_dir);
	}
	else if (strcmp(token,"RawDataFile:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%s",prms.raw_data_file);
	}
	else if (strcmp(token,"Nrows:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%i",&prms.n_rows);
	}
	else if (strcmp(token,"CollSlicewidth:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%f",&prms.coll_slicewidth);
	}
	else if (strcmp(token,"StartPos:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%f",&prms.start_pos);
	}
	else if (strcmp(token,"EndPos:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%f",&prms.end_pos);
	}
	else if (strcmp(token,"SliceThickness:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%f",&prms.slice_thickness);
	}
	else if (strcmp(token,"AcqFOV:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%f",&prms.acq_fov);
	}
	else if (strcmp(token,"ReconFOV:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%f",&prms.recon_fov);
	}
	else if (strcmp(token,"ReconKernel:")==0){
	    token=strtok(NULL,"\t\n");
	    sscanf(token,"%d",&prms.recon_kernel);
	}
	else if (strcmp(token,"Readings:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%d",&prms.n_readings); 
 	} 
 	else if (strcmp(token,"Xorigin:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%f",&prms.x_origin); 
 	} 
 	else if (strcmp(token,"Yorigin:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%f",&prms.y_origin); 
 	} 
 	else if (strcmp(token,"Zffs:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%i",&prms.z_ffs); 
 	} 
 	else if (strcmp(token,"Phiffs:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%i",&prms.phi_ffs); 
 	} 
 	else if (strcmp(token,"Scanner:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%i",&prms.scanner); 
 	} 
 	else if (strcmp(token,"FileType:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%i",&prms.file_type); 
 	} 
 	else if (strcmp(token,"RawOffset:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%i",&prms.raw_data_offset); 
 	} 
 	else if (strcmp(token,"Nx:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%i",&prms.nx); 
 	} 
 	else if (strcmp(token,"Ny:")==0){ 
 	    token=strtok(NULL,"\t\n"); 
 	    sscanf(token,"%i",&prms.ny); 
 	} 
 	else { 
 	    token=strtok(NULL,"\t\n"); 
 	} 

 	token=strtok(NULL,"\t\n"); 
     } 

     free(prm_buffer); 
     return prms; 
 } 

struct ct_geom configure_ct_geom(struct recon_params rp){ 
    struct ct_geom cg; 

    switch (rp.scanner){ 

    case 0: // Non-standard scanner (in this case Fred Noo's Simulated Scanner)

	//float det_spacing_1=1.4083f;
	//float det_spacing_2=1.3684f;
	 
	// Physical geometry of the scanner (cannot change from scan to scan) 
	cg.r_f=570.0f; 
	cg.src_to_det=1040.0f; 
	cg.anode_angle=7.0f*pi/180.0f; 
	cg.fan_angle_increment=1.4083f/cg.src_to_det;
	cg.theta_cone=2.0f*atan(7.5f*1.3684f/cg.src_to_det);
	cg.central_channel=335.25f; 

	// Size and setup of the detector helix 
	cg.n_proj_turn=1160; 
	cg.n_proj_ffs=cg.n_proj_turn*pow(2,rp.phi_ffs)*pow(2,rp.z_ffs); 
	cg.n_channels=672; 
	cg.n_channels_oversampled=2*cg.n_channels; 
	cg.n_rows=(unsigned int)rp.n_rows; 
	cg.n_rows_raw=(unsigned int)(rp.n_rows/pow(2,rp.z_ffs)); 
	cg.z_rot=15.6f; // This is related to the pitch, which we'll have to figure out 
	cg.add_projections=(cg.fan_angle_increment*cg.n_channels/2)/(2.0f*pi/cg.n_proj_turn)+10; 
	cg.add_projections_ffs=cg.add_projections*pow(2,rp.z_ffs)*pow(2,rp.phi_ffs); 

	break; 

    case 1: // Definition AS 
	
	// Physical geometry of the scanner (cannot change from scan to scan) 
	cg.r_f=595.0f; 
	cg.src_to_det=1085.6f; 
	cg.anode_angle=7.0f*pi/180.0f; 
	cg.fan_angle_increment=0.067864f*pi/180.0f; 
	cg.theta_cone=2.0f*atan(7.5f*1.2f/cg.r_f); 
	cg.central_channel=366.25f;

	// Size and setup of the detector helix 
	cg.n_proj_turn=1152; 
	cg.n_proj_ffs=cg.n_proj_turn*pow(2,rp.phi_ffs)*pow(2,rp.z_ffs); 
	cg.n_channels=736; 
	cg.n_channels_oversampled=2*cg.n_channels; 
	cg.n_rows=(unsigned int)rp.n_rows; 
	cg.n_rows_raw=(unsigned int)(rp.n_rows/pow(2,rp.z_ffs)); 
	cg.z_rot=19.2f; // This is related to the pitch, which we'll have to figure out 
	cg.add_projections=(cg.fan_angle_increment*cg.n_channels/2)/(2.0f*pi/cg.n_proj_turn)+10; 
	cg.add_projections_ffs=cg.add_projections*pow(2,rp.z_ffs)*pow(2,rp.phi_ffs); 
	
	break; 

    case 2: // Sensation 64 

 	// Physical geometry of the scanner (cannot change from scan to scan) 
 	cg.r_f=570.0f; 
 	cg.src_to_det=1040.0f; 
 	cg.anode_angle=12.0f*pi/180.0f; 
 	cg.fan_angle_increment=0.07758621f*pi/180.0f; 
 	cg.theta_cone=2.0f*atan(7.5f*1.2f/cg.r_f); 
 	cg.central_channel=334.25f; 

 	// Size and setup of the detector helix 
 	cg.n_proj_turn=1160; 
 	cg.n_proj_ffs=cg.n_proj_turn*pow(2,rp.phi_ffs)*pow(2,rp.z_ffs); 
 	cg.n_channels=672; 
 	cg.n_channels_oversampled=2*cg.n_channels; 
 	cg.n_rows=(unsigned int)rp.n_rows; 
 	cg.n_rows_raw=(unsigned int)(rp.n_rows/pow(2,rp.z_ffs)); 
 	cg.z_rot=19.2f; // This is related to the pitch, which we'll have to figure out 
 	cg.add_projections=(cg.fan_angle_increment*cg.n_channels/2)/(2.0f*pi/cg.n_proj_turn)+10; 
 	cg.add_projections_ffs=cg.add_projections*pow(2,rp.z_ffs)*pow(2,rp.phi_ffs); 

 	break; 
    } 

    cg.acq_fov=rp.acq_fov; 

    if (rp.phi_ffs==1){
	cg.central_channel=floor(cg.central_channel)+0.375f;
	//cg.central_channel+=0.375f; 
    }
    

    return cg;
}

void configure_reconstruction(struct recon_metadata *mr){
    /* --- Get working directory and User's home directory --- */
    struct passwd *pw=getpwuid(getuid());
    const char * homedir=pw->pw_dir;
    strcpy(mr->homedir,homedir);
    getcwd(mr->install_dir,4096*sizeof(char));

    /* --- Get tube angles and table positions --- */
    struct ct_geom cg=mr->cg;
    struct recon_params rp=mr->rp;

    // Allocate the memory
    mr->tube_angles=(float*)calloc(rp.n_readings,sizeof(float));
    mr->table_positions=(double*)calloc(rp.n_readings,sizeof(double));
    
    strcat(rp.raw_data_dir,rp.raw_data_file);
    
    FILE * raw_file;
    raw_file=fopen(rp.raw_data_dir,"rb");
    if (raw_file==NULL){
	perror("Raw data file not found.");
	exit(1);	
    }

    switch (rp.file_type){
    case 0:{; // Binary file
	for (int i=0;i<rp.n_readings;i++){
	    mr->tube_angles[i]=fmod(((360.0f/cg.n_proj_ffs)*i),360.0f);
	    mr->table_positions[i]=(rp.n_readings/cg.n_proj_ffs)*cg.z_rot-i*cg.z_rot/cg.n_proj_ffs;
	}	
	break;}
    case 1:{; //PTR
	for (int i=0;i<rp.n_readings;i++){
	    mr->tube_angles[i]=ReadPTRTubeAngle(raw_file,i,cg.n_channels,cg.n_rows_raw);
	    mr->table_positions[i]=(double)ReadPTRTablePosition(raw_file,i,cg.n_channels,cg.n_rows_raw)/1000.0;
	}
	break;}
    case 2:{; //CTD
	for (int i=0;i<rp.n_readings;i++){
	    mr->tube_angles[i]=ReadCTDTubeAngle(raw_file,i,cg.n_channels,cg.n_rows_raw);
	    mr->table_positions[i]=(double)ReadCTDTablePosition(raw_file,i,cg.n_channels,cg.n_rows_raw)/1000.0;
	}
	break;}
    case 3:{; //IMA
	int raw_data_subtype=mr->rp.scanner; // Determine if we're looking for PTR or CTD
	
	for (int i=0;i<rp.n_readings;i++){
	    mr->tube_angles[i]=ReadIMATubeAngle(raw_file,i,cg.n_channels,cg.n_rows_raw,raw_data_subtype,rp.raw_data_offset);
	    mr->table_positions[i]=(double)ReadIMATablePosition(raw_file,i,cg.n_channels,cg.n_rows_raw,raw_data_subtype,rp.raw_data_offset)/1000.0;
	}
	break;}
    }
    fclose(raw_file);
    
    /* --- Figure out how many and which projections to grab --- */
    int n_ffs=pow(2,rp.z_ffs)*pow(2,rp.phi_ffs);
    int n_slices_block=BLOCK_SLICES;
    
    int recon_direction=fabs(rp.end_pos-rp.start_pos)/(rp.end_pos-rp.start_pos);
    if (recon_direction!=1&&recon_direction!=-1) // user request one slice (end_pos==start_pos)
	recon_direction=1;

    int n_slices_requested=floor(fabs(rp.end_pos-rp.start_pos)/rp.slice_thickness)+1;
    int n_slices_recon=(n_slices_requested-1)+(32-(n_slices_requested-1)%32);

    int n_blocks=n_slices_recon/n_slices_block;
    
    float recon_start_pos=rp.start_pos;
    float recon_end_pos=rp.start_pos+recon_direction*(n_slices_recon-1)*rp.slice_thickness;
    int array_direction=fabs(mr->table_positions[100]-mr->table_positions[0])/(mr->table_positions[100]-mr->table_positions[0]);
    int idx_slice_start=array_search(recon_start_pos,mr->table_positions,rp.n_readings,array_direction);
    int idx_slice_end=array_search(recon_end_pos,mr->table_positions,rp.n_readings,array_direction);

    // We always pull projections in the order they occur in the raw
    // data.  If the end_pos comes before the start position in the
    // array, we use the end_pos as the "first" slice to pull
    // projections for.  This method will take into account the
    // ordering of projections with ascending or descending table
    // position, as well as any slice ordering the user requests.
    
    int idx_pull_start;
    int idx_pull_end;
    if (idx_slice_start>idx_slice_end){
	idx_pull_start=idx_slice_end-cg.n_proj_ffs/2-cg.add_projections_ffs;
	idx_pull_start=(idx_pull_start-1)+(n_ffs-(idx_pull_start-1)%n_ffs);
	idx_pull_end=idx_slice_start+cg.n_proj_ffs/2+cg.add_projections_ffs;
	idx_pull_end=(idx_pull_end-1)+(n_ffs-(idx_pull_end-1)%n_ffs);
    }
    else{
	idx_pull_start=idx_slice_start-cg.n_proj_ffs/2-cg.add_projections_ffs;
	idx_pull_start=(idx_pull_start-1)+(n_ffs-(idx_pull_start-1)%n_ffs);
	idx_pull_end=idx_slice_end+cg.n_proj_ffs/2+cg.add_projections_ffs;
	idx_pull_end=(idx_pull_end-1)+(n_ffs-(idx_pull_end-1)%n_ffs);
    }

    idx_pull_end+=256;
   
    int n_proj_pull=idx_pull_end-idx_pull_start;

    // Ensure that we have a number of projections divisible by 128 (because GPU)
    n_proj_pull=(n_proj_pull-1)+(128-(n_proj_pull-1)%128);
    idx_pull_end=idx_pull_start+n_proj_pull;
    
    // copy this info into our recon metadata
    mr->ri.n_ffs=n_ffs;
    mr->ri.n_slices_requested=n_slices_requested;
    mr->ri.n_slices_recon=n_slices_recon;
    mr->ri.n_slices_block=n_slices_block;
    mr->ri.n_blocks=n_blocks;
    mr->ri.idx_slice_start=idx_slice_start;
    mr->ri.idx_slice_end=idx_slice_end; 
    mr->ri.recon_start_pos=recon_start_pos;
    mr->ri.recon_end_pos=recon_end_pos;;
    mr->ri.idx_pull_start=idx_pull_start;
    mr->ri.idx_pull_end=idx_pull_end;
    mr->ri.n_proj_pull=n_proj_pull;

    /* --- Allocate our raw data array and our rebin array --- */
    mr->ctd.raw=(float*)calloc(cg.n_channels*cg.n_rows_raw*n_proj_pull,sizeof(float));
    mr->ctd.rebin=(float*)calloc(cg.n_channels_oversampled*cg.n_rows*(n_proj_pull-2*cg.add_projections_ffs)/n_ffs,sizeof(float));
    mr->ctd.image=(float*)calloc(rp.nx*rp.ny*n_slices_recon,sizeof(float));
}

void update_block_info(recon_metadata *mr){

    struct recon_info ri=mr->ri;
    struct recon_params rp=mr->rp;
    struct ct_geom cg=mr->cg;
    
    /* --- Figure out how many and which projections to grab --- */
    int n_ffs=pow(2,rp.z_ffs)*pow(2,rp.phi_ffs);

    int recon_direction=fabs(rp.end_pos-rp.start_pos)/(rp.end_pos-rp.start_pos);
    if (recon_direction!=1&&recon_direction!=-1) // user request one slice (end_pos==start_pos)
	recon_direction=1;
    
    float block_slice_start=ri.recon_start_pos+recon_direction*ri.cb.block_idx*rp.slice_thickness*ri.n_slices_block;
    float block_slice_end=block_slice_start+recon_direction*(ri.n_slices_block-1)*rp.slice_thickness;
    int array_direction=fabs(mr->table_positions[100]-mr->table_positions[0])/(mr->table_positions[100]-mr->table_positions[0]);
    int idx_block_slice_start=array_search(block_slice_start,mr->table_positions,rp.n_readings,array_direction);
    int idx_block_slice_end=array_search(block_slice_end,mr->table_positions,rp.n_readings,array_direction);

    // We always pull projections in the order they occur in the raw
    // data.  If the end_pos comes before the start position in the
    // array, we use the end_pos as the "first" slice to pull
    // projections for.  This method will take into account the
    // ordering of projections with ascending or descending table
    // position, as well as any slice ordering the user requests.
    
    int idx_pull_start;
    int idx_pull_end;
    if (idx_block_slice_start>idx_block_slice_end){
	idx_pull_start=idx_block_slice_end-cg.n_proj_ffs/2-cg.add_projections_ffs;
	idx_pull_start=(idx_pull_start-1)+(n_ffs-(idx_pull_start-1)%n_ffs);
	idx_pull_end=idx_block_slice_start+cg.n_proj_ffs/2+cg.add_projections_ffs;
	idx_pull_end=(idx_pull_end-1)+(n_ffs-(idx_pull_end-1)%n_ffs);
    }
    else{
	idx_pull_start=idx_block_slice_start-cg.n_proj_ffs/2-cg.add_projections_ffs;
	idx_pull_start=(idx_pull_start-1)+(n_ffs-(idx_pull_start-1)%n_ffs);
	idx_pull_end=idx_block_slice_end+cg.n_proj_ffs/2+cg.add_projections_ffs;
	idx_pull_end=(idx_pull_end-1)+(n_ffs-(idx_pull_end-1)%n_ffs);
    }

    idx_pull_end+=256;
   
    int n_proj_pull=idx_pull_end-idx_pull_start;

    // Ensure that we have a number of projections divisible by 128 (because GPU)
    n_proj_pull=(n_proj_pull-1)+(128-(n_proj_pull-1)%128);
    idx_pull_end=idx_pull_start+n_proj_pull;
    
    // copy this info into our recon metadata
    mr->ri.cb.block_slice_start=block_slice_start;
    mr->ri.cb.block_slice_end=block_slice_end;
    mr->ri.cb.idx_block_slice_start=idx_block_slice_start;
    mr->ri.cb.idx_block_slice_end=idx_block_slice_end; 

    mr->ri.idx_pull_start=idx_pull_start;
    mr->ri.idx_pull_end=idx_pull_end;
    mr->ri.n_proj_pull=n_proj_pull;

    mr->ri.cb.block_idx++;
}

void extract_projections(struct recon_metadata * mr){

    float * frame_holder=(float*)calloc(mr->cg.n_channels*mr->cg.n_rows_raw,sizeof(float));

    FILE * raw_file;
    struct recon_params rp=mr->rp;
    struct ct_geom cg=mr->cg;
    strcat(rp.raw_data_dir,rp.raw_data_file);
    raw_file=fopen(rp.raw_data_dir,"rb");
    
    switch (mr->rp.file_type){
    case 0:{ // binary
	for (int i=0;i<mr->ri.n_proj_pull;i++){
	    ReadBinaryFrame(raw_file,mr->ri.idx_pull_start+i,cg.n_channels,cg.n_rows_raw,frame_holder);
	    for (int j=0;j<cg.n_channels*cg.n_rows_raw;j++){
		mr->ctd.raw[j+cg.n_channels*cg.n_rows_raw*i]=frame_holder[j];
	    }
	}
	break;}
    case 1:{ // PTR
	for (int i=0;i<mr->ri.n_proj_pull;i++){
	    ReadPTRFrame(raw_file,mr->ri.idx_pull_start+i,cg.n_channels,cg.n_rows_raw,frame_holder);
	    for (int j=0;j<cg.n_channels*cg.n_rows_raw;j++){
		mr->ctd.raw[j+cg.n_channels*cg.n_rows_raw*i]=frame_holder[j]/2294.5f;
	    }
	}
	break;}
    case 2:{ // CTD
	for (int i=0;i<mr->ri.n_proj_pull;i++){
	    ReadCTDFrame(raw_file,mr->ri.idx_pull_start+i,cg.n_channels,cg.n_rows_raw,frame_holder);
	    for (int j=0;j<cg.n_channels*cg.n_rows_raw;j++){
		mr->ctd.raw[j+cg.n_channels*cg.n_rows_raw*i]=frame_holder[j]/2294.5f;
	    }
	}
	break;}
    case 3:{ // IMA (wraps either PTR or IMA)
	int raw_data_subtype=rp.scanner;
	for (int i=0;i<mr->ri.n_proj_pull;i++){
	    ReadIMAFrame(raw_file,mr->ri.idx_pull_start+i,cg.n_channels,cg.n_rows_raw,frame_holder,raw_data_subtype,rp.raw_data_offset);
	    for (int j=0;j<cg.n_channels*cg.n_rows_raw;j++){
		mr->ctd.raw[j+cg.n_channels*cg.n_rows_raw*i]=frame_holder[j]/2294.5f;
	    }
	}
	break;}	
    }

    // Check "testing" flag, write raw to disk if set
    if (mr->flags.testing){
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/raw.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(mr->ctd.raw,sizeof(float),cg.n_channels*cg.n_rows_raw*mr->ri.n_proj_pull,outfile);
	fclose(outfile);
    }
    
    fclose(raw_file);
    free(frame_holder);
}

void finish_and_cleanup(struct recon_metadata * mr){

    struct recon_params rp=mr->rp;
    struct recon_info ri=mr->ri;

    float * temp_out=(float*)calloc(rp.nx*rp.ny*ri.n_slices_recon,sizeof(float));

    int recon_direction=(mr->ri.idx_slice_start-mr->ri.idx_slice_end)/abs(mr->ri.idx_slice_start-mr->ri.idx_slice_end);

    if (recon_direction!=1&&recon_direction!=-1&&recon_direction!=0)
	printf("An error may have occurred\n");

    if (recon_direction!=1&&recon_direction!=-1)
	recon_direction=1;

    int table_direction=(mr->table_positions[1000]-mr->table_positions[0])/abs(mr->table_positions[1000]-mr->table_positions[0]);
    if (table_direction!=1&&table_direction!=-1)
	printf("Axial scans are currently unsupported, or a different error has occurred\n");
    
    // Check for a reversed stack of images and flip, otherwise just copy
    if (recon_direction!=table_direction){
	for (int b=0;b<ri.n_blocks;b++){
	    for (int z=0;z<ri.n_slices_block;z++){
		for (int x=0;x<rp.nx;x++){
		    for (int y=0;y<rp.ny;y++){
			long block_offset=b*rp.nx*rp.ny*ri.n_slices_block;
			temp_out[z*rp.nx*rp.ny+y*rp.nx+x+block_offset]=mr->ctd.image[((ri.n_slices_block-1)-z)*rp.nx*rp.ny+y*rp.nx+x+block_offset];
		    }
		}
	    }
	}
    }
    else{
	for (int z=0;z<ri.n_slices_recon;z++){
	    for (int x=0;x<rp.nx;x++){
		for (int y=0;y<rp.ny;y++){
		    temp_out[z*rp.nx*rp.ny+y*rp.nx+x]=mr->ctd.image[z*rp.nx*rp.ny+y*rp.nx+x];
		}
	    }
	}	
    }
    
    // Write the image data to disk
    char fullpath[4096+255]={0};
    sprintf(fullpath,"%s/Desktop/%s.img",mr->homedir,mr->rp.raw_data_file);
    FILE * outfile=fopen(fullpath,"w");
    fwrite(temp_out,sizeof(float),mr->rp.nx*mr->rp.ny*mr->ri.n_slices_requested,outfile);
    fclose(outfile);

    // Check "testing" flag, if set write all reconstructed data to disk
    if (mr->flags.testing){
	char fullpath[4096+255];
	strcpy(fullpath,mr->homedir);
	strcat(fullpath,"/Desktop/image_data.ct_test");
	FILE * outfile=fopen(fullpath,"w");
	fwrite(temp_out,sizeof(float),rp.nx*rp.ny*ri.n_slices_recon,outfile);
	fclose(outfile);
    }
    
    // Free all remaining allocations in metadata
    free(mr->ctd.raw);
    free(mr->ctd.rebin);
    free(mr->ctd.image);
    free(mr->tube_angles);
    free(mr->table_positions);
    
}

int array_search(float key,double * array,int numel_array,int search_type){
    int idx=0;

    switch (search_type){
    case -1:{// Array descending
	while (key<array[idx]&&idx<numel_array){
	    idx++;}
	break;}
    case 0:{// Find where we're equal
	while (key!=array[idx]&&idx<numel_array){
	    idx++;}
	break;}
    case 1:{// Array ascending
	while (key>array[idx]&&idx<numel_array){
	    idx++;}
	break;}
    }

    return idx;
}
