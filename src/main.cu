#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <recon_structs.h>
#include <setup.h>
#include <rebin_filter.h>
#include <backproject.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
	{
	    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	    if (abort) exit(code);
	}
}

void usage(){
    printf("\n");
    printf("usage: recon [options] input_prm_file\n\n");
    printf("    Options:\n");
    printf("          -v: verbose.\n");
    printf("          -t: test files will be written to desktop.\n");
    printf("\n");
    printf("Copyright John Hoffman 2015\n\n");
    exit(0);
}


int main(int argc, char ** argv){
    
    struct recon_metadata mr;
    memset(&mr,0,sizeof(struct recon_metadata));

    // Parse any command line arguments
    if (argc<2)
	usage();
    
    for (int i=1;i<(argc-1);i++){
	if (strcmp(argv[i],"-t")==0){
	    mr.flags.testing=1;
	}
	else if (strcmp(argv[i],"-v")==0){
	    mr.flags.verbose=1;
	}
	else{
	    usage();
	}
    }

    /* --- Step 0: configure our GPU */
    // We want to send to the GPU furthest back in the list
    // which is unlikely to have a display connected
    int device_count=0;
    cudaGetDeviceCount(&device_count);
    if (mr.flags.verbose)
	printf("Working on GPU %i \n",device_count-1);
    gpuErrchk(cudaSetDevice(device_count-1));
    cudaDeviceReset();
    
    /* --- Step 1-3 handled by functions in setup.cu --- */
    // Step 1: Parse input file
    if (mr.flags.verbose)
	printf("Reading PRM file...\n");
    mr.rp=configure_recon_params(argv[argc-1]);

    // Step 2a: Setup scanner geometry
    if (mr.flags.verbose)
	printf("Configuring scanner geometry...\n");
    mr.cg=configure_ct_geom(mr.rp);
    
    // Step 2b: Configure all remaining information
    if (mr.flags.verbose)
	printf("Configuring final reconstruction parameters...\n");
    configure_reconstruction(&mr);

    // Step 3: Extract raw data from file into memory
    if (mr.flags.verbose)
	printf("Reading raw data from file...\n");
    extract_projections(&mr);
    
    /* --- Step 4 handled by functions in rebin_filter.cu --- */
    // Step 4: Rebin and filter
    if (mr.flags.verbose)
	printf("Rebinning and filtering data...\n");
    rebin_filter(&mr);
    
    /* --- Step 5 handled by functions in backproject.cu ---*/
    // Step 5: Backproject
    if (mr.flags.verbose)
	printf("Backprojecting...\n");
    backproject(&mr);
    
    // Step 6: Save image data to disk (found in setup.cu)
    if (mr.flags.verbose)
	printf("Writing image data to disk...\n");
    finish_and_cleanup(&mr);

    if (mr.flags.verbose)
	printf("Done.\n");

    cudaDeviceReset();
    return 0;
   
}
