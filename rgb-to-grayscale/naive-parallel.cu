#include <stdio.h>
#include "../utils.h"


// Naive implementation: 1 thread for each pixel
__global__ void rgb_to_grayscale_parallel(uchar4* input, unsigned char *output, int w, int h) {
    // Global coordinates of the pixel in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x; // Index of the current pixel in the linear array

    // This data type also contains alpha channel, but it is ignored
    uchar4 p = input[idx];
    float r = p.x; // Red
    float g = p.y; // Green
    float b = p.z; // Blue

    // Grayscale formula (based on the non-linear human perception of color)
    float gray = 0.3*r + 0.59*g + 0.11*b;

    output[idx] = (unsigned char) gray;
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img(IMG_PATH);
    SDL_LockSurface(image); // Lock to access pixels
    const int width = image->w;
    const int height = image->h;
    const int input_size = width * height * sizeof(uchar4);
    const int output_size = width * height * sizeof(unsigned char);
    
    // Output image
    unsigned char* grayscale_image = (unsigned char*) malloc(width * height * sizeof(unsigned char));

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    
    // Allocate GPU memory
    // Input
    uchar4 *d_pixels_input;
    CUDA_CHECK(cudaMalloc(&d_pixels_input, input_size));
    //Output
    unsigned char *d_pixels_output;
    CUDA_CHECK(cudaMalloc(&d_pixels_output, output_size));

    dim3 block(32, 32); // 32x32 = 1024 threads per block (max for 4080 SUPER)
    // Works for every image resolution
    dim3 grid((width + block.x - 1) / block.x, // 240
              (height + block.y - 1) / block.y); // 135

    // HtoD
    CUDA_CHECK(cudaMemcpy(d_pixels_input, image->pixels, input_size, cudaMemcpyHostToDevice));

    rgb_to_grayscale_parallel<<<grid, block>>>(d_pixels_input, d_pixels_output, width, height); // Run the kernel
    CUDA_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    // DtoH, copy back the results
    CUDA_CHECK(cudaMemcpy(grayscale_image, d_pixels_output, output_size, cudaMemcpyDeviceToHost)); // Implicit barrier

    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Total CUDA time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // -------------------------------------------------- END CUDA --------------------------------------------------

    show_image(grayscale_image, width, height, "Conversione da RGB a scala di grigi - Versione parallela naive");

    // ----- Cleanup -----
    CUDA_CHECK(cudaFree(d_pixels_input));
    CUDA_CHECK(cudaFree(d_pixels_output));
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
}