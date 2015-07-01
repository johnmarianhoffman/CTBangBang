# CTBangBang

## What is CTBangBang?

CTBangBang is lightweight, gpu-based, reconstruction software for third generation helical CT scans.  It is intended to be fast, adaptable, and (relatively) easy to use and understand.  It is free software (released under the GNU GPLv2) intended for use in research and education.  Modification and contributions are encouraged.

CTBangBang is an implementation of the algorithms outlined in the following publications:

K. Stierstorfer, A. Rauscher, J. Boese, H. Bruder, S. Schaller, and T. Flohr, “Weighted FBP—a simple approximate 3D FBP algorithm for multislice spiral CT with good dose usage for arbitrary pitch,” Phys. Med. Biol., vol. 49, no. 11, pp. 2209–2218, Jun. 2004.

T. G. Flohr, K. Stierstorfer, S. Ulzheimer, H. Bruder, a. N. Primak, and C. H. McCollough, “Image reconstruction and image quality evaluation for a 64-slice CT scanner with z-flying focal spot,” Med. Phys., vol. 32, no. 8, p. 2536, 2005.

with influences from:

T. Zinßer and B. Keck, “Systematic Performance Optimization of Cone-Beam Back-Projection on the Kepler Architecture,” Fully Three-Dimensional - Image Reconstr. Radiol. Nucl. Med., pp. 225–228, 2013.

## What it is not

### CTBangBang is not what the manufacturers use

While our algorithms are based off of publications that are perhaps relevant to some of the current algorithms used in industry, they are not the algorithms used on clinical CT scanners and we make no claims to the similarity between our reconstructed images and what is arrived at clinically.

Work has been done to objectively evaluate the quality of our reconstructions.  This can be found here:

(insert link to technical note and/or whitepaper)

### CTBangBang is not a library

There are many great reconstruction libraries out there (http://conrad.stanford.edu/, http://www.openrtk.org/ to name two), and perhaps one day CTBangBang will be recast as a library.  Currently however, it is not a library, it a program.

It is structured modularly, so that major subsections of the reconstruction process are easy to identify/customize/edit/etc. so there are library-like qualities to the project to make it easy to use.

CTBangBang is designed to be compiled and run to reconstruct projection data from start to finish.

## Versions

The latest working version can be found on GitHub at https://github.com/captnjohnny1618/CTBangBang

Bleeding edge updates can be found at https://github.com/captnjohnny1618/CTBangBang/tree/develop

## Installation

(to be added later)

## Use

(to be added later)

## License

GNU GPLv2

(more info to be added later)

Copyright 2015 John Hoffman