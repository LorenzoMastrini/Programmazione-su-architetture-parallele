// Naive implementation: 1 thread for each pixel

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include "../utils.h"

// We work with 8K images, made of 7680x4320 pixels
#define WIDTH 7680
#define HEIGHT 4320
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)



__global__ void rgb_to_grayscale_parallel(Uint32* dev_pixels, int dev_w, int dev_h) {
    // Global coordinates of the pixel in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dev_w || y >= dev_h) return;

    int idx = y * dev_w + x; // Index of the current pixel in the linear array

    // This data type also contains alpha channel, but it is ignored
    Uint32 p = dev_pixels[idx];
    Uint8 r = (p >> 16) & 0xFF; // Red channel
    Uint8 g = (p >> 8) & 0xFF; // Green channel
    Uint8 b = p & 0xFF; // Blue channel
    Uint8 gray = 0.3*r + 0.59*g + 0.11*b; // Formula based on the non-linear human perception of color

    dev_pixels[idx] = (gray << 16) | (gray << 8) | gray;
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");

    int size = WIDTH * HEIGHT * sizeof(Uint32);
    
    // profiling
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocate GPU memory
    Uint32 *d_pixels;
    HANDLE_ERROR(cudaMalloc(&d_pixels, size));
    HANDLE_ERROR(cudaMemcpy(d_pixels, image->pixels, size, cudaMemcpyHostToDevice));

    dim3 block(32, 32); // 32x32 = 1024 threads per block (max for 4080 SUPER)
    // Works for every image resolution
    dim3 grid((WIDTH + block.x - 1) / block.x, // 240
              (HEIGHT+ block.y - 1) / block.y); // 135

    rgb_to_grayscale_parallel<<<grid, block>>>(d_pixels, WIDTH, HEIGHT); // Run the kernel
    checkCUDAError("Errore nella chiamata al kernel 'rgb_to_grayscale_parallel()'");

    // Copy back the results
    HANDLE_ERROR(cudaMemcpy(image->pixels, d_pixels, size, cudaMemcpyDeviceToHost)); // implicit barrier

    // profiling
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel time: %.3f ms\n", time);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    

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
    HANDLE_ERROR(cudaFree(d_pixels));
    SDL_FreeSurface(image);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}