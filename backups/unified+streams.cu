#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include "../utils.h"

// We work with 8K images, made of 7680x4320 pixels
#define WIDTH 7680
#define HEIGHT 4320
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)
#define BLOCK_X 32
#define BLOCK_Y 16
#define N_STREAMS 32 // Create 2 streams



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
    int id;
    cudaGetDevice(&id);

    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");

    uchar4 *pixels; // Host pointer to image pixels
    int size = WIDTH * HEIGHT * sizeof(uchar4);
    HANDLE_ERROR(cudaMallocManaged(&pixels, size)); // Allocate unified memory

    SDL_Surface* image_um = SDL_CreateRGBSurfaceWithFormatFrom(
        pixels,
        WIDTH, HEIGHT, 32, WIDTH * 4,
        SDL_PIXELFORMAT_RGBA32
    );

    SDL_BlitSurface(image, NULL, image_um, NULL);
    SDL_FreeSurface(image);


    // profiling
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // CUDA streams
    const int rows_per_stream = HEIGHT / N_STREAMS;
    cudaStream_t streams[N_STREAMS];

    for(int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);

        int start_row = i * rows_per_stream;
        uchar4 *slice_ptr = pixels + start_row * WIDTH;
        size_t slice_size = rows_per_stream * WIDTH * sizeof(uchar4);

        cudaMemPrefetchAsync(slice_ptr, slice_size, id, streams[i]);

        dim3 block(32, 16); // 32x32 = 1024 threads per block (max for 4080 SUPER)
        // Works for every image resolution
        dim3 grid((WIDTH + block.x - 1) / block.x, // 240
                (rows_per_stream + block.y - 1) / block.y); // 135

        rgb_to_grayscale_parallel<<<grid, block, 0, streams[i]>>>(slice_ptr, WIDTH, rows_per_stream); // Run the kernel
        //checkCUDAError("Errore nella chiamata al kernel 'rgb_to_grayscale_parallel()'");
        checkCUDAError((char*) cudaGetLastError());

        cudaMemPrefetchAsync(slice_ptr, slice_size, cudaCpuDeviceId, streams[i]);
    }

    // Wait for all streams
    for (int i = 0; i < N_STREAMS; i++)
        cudaStreamSynchronize(streams[i]);

    // profiling
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Kernel time: %.3f ms\n", time);

    // Create the window (fullscreen)
    SDL_Window *window = create_window("RGB to grayscale - Optimized parallel");

    // Get the surface of the window (where we will draw)
    SDL_Surface *screen = SDL_GetWindowSurface(window);
    if (!screen) {
        printf("Errore SDL_GetWindowSurface: %s\n", SDL_GetError());
        return 1;
    }

    // Copy the surface in the window (correctly scaling the dimensions)
    SDL_BlitScaled(image_um, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);

    SDL_Delay(SHOW_IMAGE_FOR); // Show for 3 seconds

    // Cleanup
    for (int i = 0; i < N_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
    cudaFree(pixels);
    SDL_FreeSurface(image_um);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}