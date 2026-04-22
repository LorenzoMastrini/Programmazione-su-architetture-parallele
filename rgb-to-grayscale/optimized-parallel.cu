#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include "../utils.h"

// Block dimensions fine-tuned to optimize occupancy
#define BLOCK_X 32
#define BLOCK_Y 8
#define N_STREAMS 8
#define NUM_SLICES 8 // Image height must be divisible by NUM_SLICES
#define PIXELS_PER_THREAD 4


// Use constant memory to access constants to be used in the kernel
__constant__ float GRAY_COEF[3] = {0.213f, 0.715f, 0.072f};

__device__ __forceinline__ unsigned char to_gray(const uchar4 p) {
    // fmaf(a, b, c): (a x b) + c
    return (unsigned char) fmaf(GRAY_COEF[2], p.z, fmaf(GRAY_COEF[1], p.y, GRAY_COEF[0] * p.x));
}

// Ogni thread elabora PIXELS_PER_THREAD pixel consecutivi in memoria (row-major).
// Il `grid-stride loop` è su un unico indice lineare, così `grid` e `block` possono essere 1D.
__global__ void rgb_to_grayscale_parallel(const uchar4* __restrict__ input,
                                          unsigned char* __restrict__ output,
                                          int slice_width,
                                          int slice_height) {
    const int total_pixels = slice_width * slice_height;

    // Grid-stride loop su indice lineare, ma distribuendo i PIXELS_PER_THREAD
    // con passo = blockDim.x per evitare overlap tra thread.
    const int base0 = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
    const int stride = gridDim.x * blockDim.x * PIXELS_PER_THREAD;

    for (int base = base0; base < total_pixels; base += stride) {
        #pragma unroll
        for (int i = 0; i < PIXELS_PER_THREAD; i++) {
            const int pixel_idx = base + i * blockDim.x;
            if (pixel_idx < total_pixels)
                output[pixel_idx] = to_gray(input[pixel_idx]);
        }
    }
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img(IMG_PATH);
    SDL_LockSurface(image); // Lock to access pixels
    const int width = image->w;
    const int height = image->h;
    const int input_size = width * height * sizeof(uchar4);
    const int output_size = width * height * sizeof(unsigned char);

    // Register existing buffer as page-locked (pinned) memory
    CUDA_CHECK(cudaHostRegister(image->pixels, input_size, cudaHostRegisterPortable));
    unsigned char *h_output; // Output image
    CUDA_CHECK(cudaMallocHost(&h_output, output_size));

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Allocate GPU memory
    uchar4 *d_pixels_in;
    unsigned char *d_pixels_out;
    CUDA_CHECK(cudaMalloc(&d_pixels_in, input_size));
    CUDA_CHECK(cudaMalloc(&d_pixels_out, output_size)); // 1 byte per pixel

    // CUDA streams
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Number of rows in each slide of the the image, each handled by a stream
    const int rows_per_slice = height / NUM_SLICES;    

    // Get the number of Streaming Multiprocessor of the GPU
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    dim3 block(256);
    dim3 grid(32 * numSMs); // Optimize occupacy

    for (int i = 0; i < NUM_SLICES; i++) {
        int s = i % N_STREAMS;
        int start_row = i * rows_per_slice;

        size_t in_size  = rows_per_slice * width * sizeof(uchar4);
        size_t out_size = rows_per_slice * width * sizeof(unsigned char);

        uchar4 *slice_in  = d_pixels_in  + start_row * width;
        unsigned char *slice_out = d_pixels_out + start_row * width;
        
        // HtoD
        CUDA_CHECK(cudaMemcpyAsync(slice_in,
                                   (uchar4*) image->pixels + start_row * width,
                                   in_size,
                                   cudaMemcpyHostToDevice,
                                   streams[s]));

        // Kernel
        rgb_to_grayscale_parallel<<<grid, block, 0, streams[s]>>>(
            slice_in, slice_out, width, rows_per_slice);
        CUDA_CHECK(cudaGetLastError());

        // DtoH
        CUDA_CHECK(cudaMemcpyAsync(h_output + start_row * width,
                                   slice_out,
                                   out_size,
                                   cudaMemcpyDeviceToHost,
                                   streams[s]));
    }
    
    // Wait for all streams to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Total CUDA time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));  
    // -------------------------------------------------- END CUDA --------------------------------------------------

    show_image(h_output, width, height, "Conversione da RGB a scala di grigi - Versione parallela ottimizzata");

    // ----- Cleanup -----
    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFree(d_pixels_in));
    CUDA_CHECK(cudaFree(d_pixels_out));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaHostUnregister(image->pixels)); // Unregister page-locked (pinned) memory
    SDL_UnlockSurface(image);
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
}