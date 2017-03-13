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

#ifndef backproject_meta_h
#define backproject_meta_h

#define pi 3.1415368979f
#define K 1
#define I 1

__constant__ struct ct_geom d_cg;
__constant__ struct recon_params d_rp;

texture<float,cudaTextureType2D,cudaReadModeElementType> tex_a;
texture<float,cudaTextureType2D,cudaReadModeElementType> tex_b;

__device__ inline float W(float q){
    float out;
    float Q=0.6f;
    if (fabsf(q)<Q){
	out=1.0f;
    }
    else if ((fabsf(q)>=Q)&&(fabsf(q)<1.0f)){
	out=powf(cosf((pi/2.0f)*(fabsf(q)-Q)/(1.0f-Q)),2.0f);
    }
    else {
	out=0.0f;
    }
    return out;
}

#endif
