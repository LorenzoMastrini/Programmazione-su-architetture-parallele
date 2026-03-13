#include <stdio.h>
#include "utils.h"


int main(void) {
    cudaDeviceProp prop;
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count)); // get the number of available GPUs

    // Loop om all the GPUs
    for(int i = 0; i < count; i++) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i)); // Get GPU properties
        printf("--- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        //printf("Clock rate: %d\n", prop.clockRate);
        
        /*
        printf("Device copy overlap: ");
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
            printf( "Kernel execution timeout : " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );
        */

        printf("\n--- Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);
        printf("Async engine count: %d\n", prop.asyncEngineCount);

        printf("\n--- MP Information for device %d ---\n", i);
        printf("Streaming Multiprocessor (SM) count: %d\n", prop.multiProcessorCount);
        printf("Shared memory per SM: %ld\n", prop.sharedMemPerMultiprocessor);
        printf("Shared memory per block: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per SM: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max thread dimensions: (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}