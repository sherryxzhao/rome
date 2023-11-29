# First Step: conda data storage renavigate

1. Use `conda info` to get the **user config file** path (mine: /home/hice1/yyan376/.condarc)
2. Use text editor to edit the config file to the content of **.condarc**
3. After this step you can go ahead and remove the **.conda** folder in your head node (this folder is hidden, use `du -sh .[!.]* * | sort -rh` to see all folder size)

## Second Step: Modify **scripts/setup_conda.sh**

1. Get CUDA_DIR: request the GPU you need and contain the code `echo $CUDA_HOME` in the shell code. Submit it and see the output for CUDA directory. (I use V100, the directory is: `/usr/local/pace-apps/spack/packages/linux-rhel7-x86_64/gcc-4.8.5/cuda-11.6.0-u4jzhgn5buvcnkwuqrep25mluzkhzi3j`)
2. Substitute this path in the **scripts/setup_conda.sh** CUDA_DIR variable

## Third Step: Modify **scripts/rome.yml** and **experiments/evaluate.py**
Just use the updated version of these two files

## Last Step: The batch submission script

Refer to helloexample.sbatch

## Notes

Try running the code in Jupyter and estimate the total running time. For a 3hr test run last night, 821/1000 examples are finished. Thus in this version of helloexample.sbatch I changed the running time to 4hrs