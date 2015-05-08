// Non-GPU functions and data types to configure recontructions 
#include <recon_structs.h>

#ifndef setup_h
#define setup_h

// Step 1-3 functions
struct recon_params configure_recon_params(char * filename);
struct ct_geom configure_ct_geom(struct recon_params rp);
void configure_reconstruction(struct recon_metadata *mr);
void extract_projections(struct recon_metadata * mr);
void finish_and_cleanup(struct recon_metadata * mr);

#endif
