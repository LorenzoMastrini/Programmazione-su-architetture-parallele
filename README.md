# Programmazione su architetture parallele
Exam project that aims to parallelize various Comupter Vision algorithms using C and CUDA

# Compiling guide
To compile the project you need to modify the make file by changing the compute capability of your machine in the last two flags at row 32.

Example: CC 8.9 corresponds to flags -arch=compute_89 -code=sm_89
