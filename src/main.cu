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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <regex.h>
#include <cstdarg>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

#include <recon_structs.h>
#include <setup.h>
#include <rebin_filter.h>
#include <rebin_filter_cpu.h>
#include <backproject.h>
#include <backproject_cpu.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
	{
	    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	    if (abort) exit(code);
	}
}

void log(int verbosity, const char *string, ...);
void split_path_file(char** p, char** f, char *pf);

void usage(){
    printf("\n");
    printf("usage: recon [options] input_prm_file\n\n");
    printf("    Options:\n");
    printf("          -v: verbose.\n");
    printf("          -t: test files will be written to desktop.\n");
    printf("    --no-gpu: run program exclusively on CPU. Will override --device=i option.\n");
    printf("  --device=i: run on GPU device number 'i'\n");
    printf("    --timing: Display timing information for each step of the recon process\n");
    printf(" --benchmark: Writes timing data to file used by benchmarking tool\n");    
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
    
    regex_t regex_dev;
    regmatch_t regmatch_dev;
    if (regcomp(&regex_dev,"--device=*",0)!=0){
	printf("Regex didn't work properly\n");
	exit(1);
    }
    
    for (int i=1;i<(argc-1);i++){
	if (strcmp(argv[i],"-t")==0){
	    mr.flags.testing=1;
	}
	else if (strcmp(argv[i],"-v")==0){
	    mr.flags.verbose=1;
	}
	else if (strcmp(argv[i],"--no-gpu")==0){
	    mr.flags.no_gpu=1;
	}
	else if (regexec(&regex_dev,argv[i],1,&regmatch_dev,0)==0){
	    mr.flags.set_device=1;
	    sscanf(argv[i],"--device=%d",&mr.flags.device_number);
	}
	else if (strcmp(argv[i],"--timing")==0){
	    mr.flags.timing=1;
	}
	else if (strcmp(argv[i],"--benchmark")==0){
	    mr.flags.benchmark=1;
	} 
	else{
	    usage();
	}
    }

    log(mr.flags.verbose,"\n-------------------------\n"
                         "|      CTBangBang       |\n"
                         "-------------------------\n\n");

    log(mr.flags.verbose,"CHECKING INPUT PARAMETERS AND CONFIGURING RECONSTRUCTION\n"
	                 "\n");
    
    /* --- Get working directory and User's home directory --- */
    struct passwd *pw=getpwuid(getuid());
    
    const char * homedir=pw->pw_dir;
    strcpy(mr.homedir,homedir);
    char full_exe_path[4096]={0};
    char * exe_path=(char*)calloc(4096,sizeof(char));
    char * exe_name=(char*)calloc(255,sizeof(char));
    readlink("/proc/self/exe",full_exe_path,4096);
    split_path_file(&exe_path,&exe_name,full_exe_path);
    strcpy(mr.install_dir,exe_path);
    mr.install_dir[strlen(mr.install_dir)-1]=0;
    
    /* --- Step 0: configure our processor (CPU or GPU) */
    // We want to send to the GPU furthest back in the list which is
    // unlikely to have a display connected.  We also check for the
    // user passing a specific device number via the command line

    int device_count=0;
    cudaGetDeviceCount(&device_count);
    if (device_count==0){
	mr.flags.no_gpu=1;
    }

    // Configure the GPU/CPU selection
    if (mr.flags.no_gpu==0){
	int device;
	if (mr.flags.set_device==1){
	    log(mr.flags.verbose,"CUDA device %d requested.\n",mr.flags.device_number);
	    log(mr.flags.verbose,"Attempting to set device... ");
	    cudaSetDevice(mr.flags.device_number);
	    cudaGetDevice(&device);
	    if (device!=mr.flags.device_number){
		printf("There was a problem setting device.\n");
	    }
	    else{
		log(mr.flags.verbose,"success!\n");
	    }
	}
	else{
	    cudaSetDevice(device_count-1);
	    cudaGetDevice(&device);
	}	
	log(mr.flags.verbose,"Working on GPU %i \n",device);
	cudaDeviceReset();
    }
    else{
	log(mr.flags.verbose,"Working on CPU\n");
    }

    // --timing cuda events
    cudaEvent_t start,stop;

    // Set up benchmarking variables and output file if requested
    char fullpath[4096+255];
    strcpy(fullpath,mr.homedir);
    strcat(fullpath,"/Desktop/.tmp_benchmark.bin");
    FILE * benchmark_file;
    if (mr.flags.benchmark){
	benchmark_file=fopen(fullpath,"a");
	fseek(benchmark_file,0,SEEK_END);
    }

    cudaEvent_t bench_master_start,bench_master_stop,bench_start,bench_stop;
    if (mr.flags.benchmark){
	cudaEventCreate(&bench_master_start);
	cudaEventCreate(&bench_master_stop);
	cudaEventRecord(bench_master_start);
    }
    
    /* --- Step 1-3 handled by functions in setup.cu --- */
    // Step 1: Parse input file
    log(mr.flags.verbose,"Reading PRM file...\n");
    mr.rp=configure_recon_params(argv[argc-1]);

    // Step 2a: Setup scanner geometry
    log(mr.flags.verbose,"Configuring scanner geometry...\n");
    mr.cg=configure_ct_geom(&mr);
    
    // Step 2b: Configure all remaining information
    log(mr.flags.verbose,"Configuring final reconstruction parameters...\n");
    configure_reconstruction(&mr);

    log(mr.flags.verbose,"Allowed recon range: %.2f to %.2f\n",mr.ri.allowed_begin,mr.ri.allowed_end);

    log(mr.flags.verbose,"\nSTARTING RECONSTRUCTION\n\n");
    
    for (int i=0;i<mr.ri.n_blocks;i++){

	update_block_info(&mr);
	
	log(mr.flags.verbose,"----------------------------\n"
                             "Working on block %d of %d \n",i+1,mr.ri.n_blocks);
	
	// Step 3: Extract raw data from file into memory
	log(mr.flags.verbose,"Reading raw data from file...\n");
	extract_projections(&mr);
    
	/* --- Step 4 handled by functions in rebin_filter.cu --- */
	// Step 4: Rebin and filter
	log(mr.flags.verbose,"Rebinning and filtering data...\n");

	if (mr.flags.timing){
	    cudaEventCreate(&start);
	    cudaEventCreate(&stop);
	    cudaEventRecord(start);
	}

	if (mr.flags.benchmark){
	    cudaEventCreate(&bench_start);
	    cudaEventCreate(&bench_stop);
	    cudaEventRecord(bench_start);
	}

	if (mr.flags.no_gpu==1)
	    rebin_filter_cpu(&mr);
	else
	    rebin_filter(&mr);

	if (mr.flags.timing){
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);
	    float milli=0.0f;
	    cudaEventElapsedTime(&milli,start,stop);
	    printf("%.2f ms to rebin & filter\n",milli);
	    cudaEventDestroy(start);
	    cudaEventDestroy(stop);
	}
	if (mr.flags.benchmark){
	    cudaEventRecord(bench_stop);
	    cudaEventSynchronize(bench_stop);
	    float milli=0.0f;
	    cudaEventElapsedTime(&milli,bench_start,bench_stop);
	    // write the benchmark data to file
	    fwrite(&milli,sizeof(float),1,benchmark_file);
	    cudaEventDestroy(bench_start);
	    cudaEventDestroy(bench_stop);
	}

	/* --- Step 5 handled by functions in backproject.cu ---*/
	// Step 5: Backproject
	log(mr.flags.verbose,"Backprojecting...\n");

	if (mr.flags.timing){
	    cudaEventCreate(&start);
	    cudaEventCreate(&stop);
	    cudaEventRecord(start);
	}

	if (mr.flags.benchmark){
	    cudaEventCreate(&bench_start);
	    cudaEventCreate(&bench_stop);
	    cudaEventRecord(bench_start);
	}

	if (mr.flags.no_gpu==1)
	    backproject_cpu(&mr);
	else
	    backproject(&mr);

	if (mr.flags.timing){
	    cudaEventRecord(stop);
	    cudaEventSynchronize(stop);
	    float milli=0.0f;
	    cudaEventElapsedTime(&milli,start,stop);
	    printf("%.2f ms to backproject\n",milli);
	    cudaEventDestroy(start);
	    cudaEventDestroy(stop);
	}
	
	if (mr.flags.benchmark){
	    cudaEventRecord(bench_stop);
	    cudaEventSynchronize(bench_stop);
	    float milli=0.0f;
	    cudaEventElapsedTime(&milli,bench_start,bench_stop);
	    // write the benchmark data to file
	    fwrite(&milli,sizeof(float),1,benchmark_file);
	    cudaEventDestroy(bench_start);
	    cudaEventDestroy(bench_stop);
	}

	
    }


    // Step 6: Save image data to disk (found in setup.cu)
    log(mr.flags.verbose,"----------------------------\n\n");
    log(mr.flags.verbose,"Writing image data to %s/Desktop/%s.img\n",mr.homedir,mr.rp.raw_data_file);
    finish_and_cleanup(&mr);

    log(mr.flags.verbose,"Done.\n");

    if (mr.flags.benchmark){
	cudaEventRecord(bench_master_stop);
	cudaEventSynchronize(bench_master_stop);
	float milli=0.0f;
	cudaEventElapsedTime(&milli,bench_master_start,bench_master_stop);
	// write the benchmark data to file
	fwrite(&milli,sizeof(float),1,benchmark_file);
	cudaEventDestroy(bench_master_start);
	cudaEventDestroy(bench_master_stop);
	fclose(benchmark_file);
    }

    cudaDeviceReset();
    return 0;
   
}

void log(int verbosity, const char *string,...){
    va_list args;
    va_start(args,string);

    if (verbosity){
	vprintf(string,args);
	va_end(args);
    } 
}

void split_path_file(char** p, char** f, char *pf) {
    char *slash = pf, *next;
    while ((next = strpbrk(slash + 1, "\\/"))) slash = next;
    if (pf != slash) slash++;
    *p = strndup(pf, slash - pf);
    *f = strdup(slash);
}
