/* FreeCT_wFBP is GPU and CPU CT reconstruction Software */
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

/* This file was automatically generated using a code-generation
   script.  Changes made directly to the .c file will likely be
   overwritten.  If you have suggested changes to the script we may
   implement them in the templates.  Please contact
   freect.project@gmail.com.*/

#include <stdio.h>
#include <string.h>

#include <parse_config.h>

size_t find_char(char * str, char delim){
    int i=0;
    while ((str[i]!=delim)&&(i<strlen(str)))
	i++;

    return i;
}

void parse_config(char * config_file, struct recon_params * structure){

    FILE * conf=fopen(config_file,"r");
    
    char raw_line[1024];
    char line_no_comments[1024];
    char * token;
	
    while (fgets(raw_line,1024,conf)!=NULL){
	// Strip comments off of the line
	size_t first_delim=find_char(raw_line,COMMENT_DELIM);
	memcpy(line_no_comments,raw_line,first_delim);	

	// Parse tokens and values
	if (strcmp(line_no_comments,"")!=0&&strcmp(line_no_comments,"\n")!=0){
	    token=strtok(line_no_comments," \t");

	    
if (strcmp(token,"RawDataDir:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%s",structure->raw_data_dir);
}
if (strcmp(token,"RawDataFile:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%s",structure->raw_data_file);
}
if (strcmp(token,"OutputDir:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%s",structure->output_dir);
}
if (strcmp(token,"OutputFile:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%s",structure->output_file);
}
if (strcmp(token,"Nrows:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->n_rows);
}
if (strcmp(token,"CollSlicewidth:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->coll_slicewidth);
}
if (strcmp(token,"StartPos:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->start_pos);
}
if (strcmp(token,"EndPos:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->end_pos);
}
if (strcmp(token,"TableFeed:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->pitch_value);
}
if (strcmp(token,"PitchValue:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->pitch_value);
}
if (strcmp(token,"SliceThickness:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->slice_thickness);
}
if (strcmp(token,"AcqFOV:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->acq_fov);
}
if (strcmp(token,"ReconFOV:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->recon_fov);
}
if (strcmp(token,"ReconKernel:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->recon_kernel);
}
if (strcmp(token,"Readings:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->n_readings);
}
if (strcmp(token,"Xorigin:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->x_origin);
}
if (strcmp(token,"Yorigin:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->y_origin);
}
if (strcmp(token,"Zffs:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->z_ffs);
}
if (strcmp(token,"Phiffs:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->phi_ffs);
}
if (strcmp(token,"Scanner:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%s",structure->scanner);
}
if (strcmp(token,"FileType:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->file_type);
}
if (strcmp(token,"FileSubType:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->file_subtype);
}
if (strcmp(token,"RawOffset:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->raw_data_offset);
}
if (strcmp(token,"Nx:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%u",&structure->nx);
}
if (strcmp(token,"Ny:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%u",&structure->ny);
}
if (strcmp(token,"TubeStartAngle:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->tube_start_angle);
}
if (strcmp(token,"AdaptiveFiltration:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%f",&structure->adaptive_filtration_s);
}
if (strcmp(token,"NSlices:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->n_slices);
}
if (strcmp(token,"TableDir:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%s",structure->table_dir_str);
}
if (strcmp(token,"TableDirInt:")==0){
    printf("Token \"%s\" detected\n",token);
    token=strtok(NULL," \t");
    sscanf(token,"%d",&structure->table_dir);
}

	}
	else{
	}

	memset(line_no_comments,0,1024);
    }

    fclose(conf);
}