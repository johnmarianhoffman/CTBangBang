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

#ifndef recon_structs_h
#define recon_structs_h

struct recon_params{
    char  raw_data_dir[4096];
    char  raw_data_file[255];
    char  output_dir[4096];
    int n_rows;
    float coll_slicewidth;
    float start_pos;
    float end_pos;
    float slice_thickness;
    float tube_start_angle;
    float pitch_value;
    float acq_fov;
    float recon_fov;
    int recon_kernel;
    float x_origin;
    float y_origin;
    int n_readings;
    int z_ffs;
    int phi_ffs;
    char scanner[4096+255];
    int file_type;
    int file_subtype;
    int raw_data_offset;
    int table_dir;
    unsigned int nx;
    unsigned int ny;
};

struct block_info{
    int block_idx;
    float block_slice_start;
    float block_slice_end;
    int idx_block_slice_start;
    int idx_block_slice_end;
};

struct recon_info{
    int n_ffs;
    float data_begin_pos;
    float data_end_pos;
    float allowed_begin;
    float allowed_end;
    int n_slices_requested;
    int n_slices_recon;
    int n_slices_block;
    int n_blocks;
    int idx_slice_start;
    int idx_slice_end; 
    float recon_start_pos;
    float recon_end_pos;
    int idx_pull_start;
    int idx_pull_end;
    int n_proj_pull;
    struct block_info cb;
};
    
struct ct_data{
    float * raw;
    float * rebin;
    float * image;
};
    
struct ct_geom{
    unsigned int n_proj_turn;
    unsigned int n_proj_ffs;
    unsigned int n_channels;
    unsigned int n_channels_oversampled;
    unsigned int n_rows;
    unsigned int n_rows_raw;
    float r_f;
    float z_rot;
    float theta_cone;
    float fan_angle_increment;
    float src_to_det;
    float anode_angle;
    float central_channel;
    float acq_fov;
    int projection_offset;
    int add_projections;
    int add_projections_ffs;
    int reverse_row_interleave;
    int reverse_channel_interleave;

    // -1 table positions decreasing (SciDirTableIn);
    //  1 table positions increasing (SciDirTableOut);
    int table_direction;
};

struct flags{
    int testing;
    int verbose;
    int no_gpu;
    int set_device;
    int device_number;
    int timing;
    int benchmark;
};
    
struct recon_metadata {
    char homedir[4096];
    char install_dir[4096];
    char output_dir[4096];
    struct flags flags;
    struct recon_params rp;
    struct recon_info ri;
    struct ct_geom cg;
    struct ct_data ctd;
    
    float * tube_angles;
    double * table_positions;
};

#endif
