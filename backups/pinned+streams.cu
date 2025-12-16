#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include "../utils.h"

// We work with 8K images, made of 7680x4320 pixels
#define WIDTH 7680
#define HEIGHT 4320
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)
#define BLOCK_X 16
#define BLOCK_Y 16
#define N_STREAMS 16



__global__ void rgb_to_grayscale_parallel(uchar4* dev_pixels, int dev_w, int dev_h) {
    // Global coordinates of the pixel in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dev_w || y >= dev_h) return;

    int idx = y * dev_w + x; // Index of the current pixel in the linear array

    uchar4 p = dev_pixels[idx]; // This data type also contains alpha channel, but it is ignored

    Uint8 gray = static_cast<unsigned char>(0.3f*p.x + 0.59f*p.y + 0.11f*p.z); // Formula based on the non-linear human perception of color

    dev_pixels[idx] = make_uchar4(gray, gray, gray, p.w);
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");


    // ----- CUDA setup and calculations -----
    int size = WIDTH * HEIGHT * sizeof(uchar4);

    SDL_LockSurface(image);
    cudaHostRegister(image->pixels, size, cudaHostRegisterPortable); // Allocate page-locked (pinned) memory in the host
    SDL_UnlockSurface(image);

    // profiling
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Allocate GPU memory
    uchar4 *d_pixels;
    HANDLE_ERROR(cudaMalloc(&d_pixels, size)); // Allocate GPU memory

    // CUDA streams
    const int rows_per_stream = HEIGHT / N_STREAMS;
    cudaStream_t streams[N_STREAMS];

    for(int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);

        int start_row = i * rows_per_stream;
        uchar4 *slice_ptr = d_pixels + start_row * WIDTH;
        size_t slice_size = rows_per_stream * WIDTH * sizeof(uchar4);

        cudaMemcpyAsync(slice_ptr, (uchar4*) image->pixels + start_row * WIDTH, slice_size, cudaMemcpyHostToDevice, streams[i]);

        dim3 block(BLOCK_X, BLOCK_Y); // 32x32 = 1024 threads per block (max for 4080 SUPER)
        // Works for every image resolution
        dim3 grid((WIDTH + block.x - 1) / block.x, // 240
                  (rows_per_stream + block.y - 1) / block.y); // 135

        rgb_to_grayscale_parallel<<<grid, block, 0, streams[i]>>>(slice_ptr, WIDTH, rows_per_stream); // Run the kernel
        checkCUDAError((char*) cudaGetLastError());

        cudaMemcpyAsync((uchar4*) image->pixels + start_row * WIDTH, slice_ptr, slice_size, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all streams
    for (int i = 0; i < N_STREAMS; i++)
        cudaStreamSynchronize(streams[i]);

    // profiling
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel time: %.3f ms\n", time);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    // CUDA cleanup
    for (int i = 0; i < N_STREAMS; i++)
        HANDLE_ERROR(cudaStreamDestroy(streams[i]));
    HANDLE_ERROR(cudaFree(d_pixels));


    // ----- Show the modified image -----
    SDL_Window *window = create_window("RGB to grayscale - Optimized parallel"); // Create the window (fullscreen)
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, image);
    SDL_FreeSurface(image); // Surface no longer needed
    SDL_RenderCopy(renderer, texture, NULL, NULL); // Copy the texture to the renderer
    SDL_RenderPresent(renderer);     // Update the screen with the rendered content
    SDL_Delay(SHOW_IMAGE_FOR); // Show for 3 seconds

    // ----- Cleanup -----
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}