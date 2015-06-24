echo "Working on medium..."
./test -v ~/Study_Data/WFBP/prms/a_ffs_medium.prm
mv ~/Desktop/image_data.txt ~/Study_Data/WFBP/image_data/j_recon/ACR_Kernels/medium.img

echo "Working on sharp..."
./test -v ~/Study_Data/WFBP/prms/a_ffs_sharp.prm
mv ~/Desktop/image_data.txt ~/Study_Data/WFBP/image_data/j_recon/ACR_Kernels/sharp.img
