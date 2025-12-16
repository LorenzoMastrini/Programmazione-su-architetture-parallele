// Naive implementation: 1 thread for each pixel

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include "../utils.h"

// We work with 8K images, made of 7680x4320 pixels
#define WIDTH 7680
#define HEIGHT 4320
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)
#define N_STREAMS 4 // Create 2 streams



__global__ void rgb_to_grayscale_parallel(uchar4* dev_pixels, int dev_w, int dev_h) {
    // Global coordinates of the pixel in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dev_w || y >= dev_h) return;

    // Coalesced access
    int idx = y * dev_w + x; // Index of the current pixel in the linear array

    // This data type also contains alpha channel, but it is ignored
    uchar4 p = dev_pixels[idx];
    Uint8 r = p.x; // Red channel
    Uint8 g = p.y; // Green channel
    Uint8 b = p.z; // Blue channel
    Uint8 gray = 0.3*r + 0.59*g + 0.11*b; // Formula based on the non-linear human perception of color

    dev_pixels[idx] = make_uchar4(gray, gray, gray, p.w);
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");

    // Allocate and initialize page-locked memory
    Uint32 *pixels; // Host pointer to image pixels
        int size = WIDTH * HEIGHT * sizeof(Uint32);
        HANDLE_ERROR(cudaHostAlloc(&pixels, size, cudaHostAllocDefault));
        memcpy(pixels, (Uint32*) image->pixels, size); // Copy pixels array to the page-locked memeory
    
    // profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate GPU memory
    uchar4 *d_pixels;
    HANDLE_ERROR(cudaMalloc(&d_pixels, size));

    // CUDA streams
    cudaStream_t streams[N_STREAMS];

    // Create N streams
    for(int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);

        int rows_per_stream = HEIGHT / N_STREAMS;
        HANDLE_ERROR(cudaMemcpyAsync(d_pixels + i*rows_per_stream*WIDTH,
                                     pixels + i*rows_per_stream*WIDTH,
                                     size / N_STREAMS,
                                     cudaMemcpyHostToDevice,
                                     streams[i]));

        dim3 block(16, 16); // 32x32 = 1024 threads per block (max for 4080 SUPER)
        // Works for every image resolution
        dim3 grid((WIDTH + block.x - 1) / block.x, // 240
                (HEIGHT+ block.y - 1) / block.y); // 135

        rgb_to_grayscale_parallel<<<grid, block>>>(d_pixels + i*rows_per_stream*WIDTH, WIDTH, rows_per_stream); // Run the kernel
        checkCUDAError("Errore nella chiamata al kernel 'rgb_to_grayscale_parallel()'");

        // Copy back the results
        HANDLE_ERROR(cudaMemcpyAsync(pixels + i*rows_per_stream*WIDTH,
                                     d_pixels + i*rows_per_stream*WIDTH,
                                     size / N_STREAMS,
                                     cudaMemcpyDeviceToHost,
                                     streams[i])); // Implicit barrier
    }

    HANDLE_ERROR(cudaFree(d_pixels));
    memcpy(image->pixels, pixels, size); // Update the modified pixels to display the image
    HANDLE_ERROR(cudaFreeHost(pixels)); // Free the host page-locked memory

    // profiling
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);

    // Create the window (fullscreen)
    SDL_Window *window = create_window("RGB to grayscale - Parallel");

    // Get the surface of the window (where we will draw)
    SDL_Surface *screen = SDL_GetWindowSurface(window);
    if (!screen) {
        printf("Errore SDL_GetWindowSurface: %s\n", SDL_GetError());
        return 1;
    }

    // Copy the surface in the window (correctly scaling the dimensions)
    SDL_BlitScaled(image, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);

    SDL_Delay(SHOW_IMAGE_FOR); // Show for 3 seconds

    // Free everything
    SDL_FreeSurface(image);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}