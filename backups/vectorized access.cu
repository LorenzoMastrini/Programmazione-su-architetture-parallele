#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include "../utils.h"

// We work with 8K images, made of 7680x4320 pixels
#define WIDTH 7680
#define HEIGHT 4320
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)
// Block dimensions fine-tuned to optimize occupancy
#define BLOCK_X 32
#define BLOCK_Y 8
#define N_STREAMS 16
#define PIXELS_PER_THREAD 4


// Use constant memory to access constants to be used in the kernel
__constant__ __device__ float gray_coef[3] = {0.2126f, 0.7152f, 0.0722f};

// each thread reads and writes 4 pixels to optimize the use of L1 cache
__global__ void rgb_to_grayscale_parallel(uchar4* __restrict__ input, uint8_t* __restrict__ output, int w, int h) { // __restric__ avoid pointer aliasing
    // Global coordinates of the pixel in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x * PIXELS_PER_THREAD + PIXELS_PER_THREAD >= w || y >= h) return;

    int idx = (y * w + x * PIXELS_PER_THREAD) / 4; // Index in the uint4 vector

    // Vectorized load (16 bytes = 4 * 4 bytes)
    union {
        uint4 v;
        uchar4 p[PIXELS_PER_THREAD];
    } in;
   
    // Reinterpret uchar4 (4 bytes) to uint4 (16 bytes) to vectorize access
    in.v = reinterpret_cast<const uint4*>(input)[idx];

    // output scalare (4 byte)
    uint8_t gray[PIXELS_PER_THREAD];

    #pragma unroll
    for(int i = 0; i < PIXELS_PER_THREAD; i++) {
        gray[i] = gray_coef[0] * in.p[i].x +  // 0.299 * red
                  gray_coef[1] * in.p[i].y +  // 0.587 * green
                  gray_coef[2] * in.p[i].z;   // 0.114 * blue
    }

    // Vectorized Store
    __stcg(&reinterpret_cast<uint32_t*>(output)[idx],
           *reinterpret_cast<uint32_t*>(&gray[0])); // write‑through
    //reinterpret_cast<uint32_t*>(output)[idx] = *reinterpret_cast<uint32_t*>(&gray[0]);
    //reinterpret_cast<uint32_t*>(output)[idx + 1] = *reinterpret_cast<uint32_t*>(&gray[4]);
    
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");


    // ----- CUDA setup and calculations -----
    int size = WIDTH * HEIGHT * sizeof(uchar4);

    SDL_LockSurface(image);
    cudaHostRegister(image->pixels, size, cudaHostRegisterPortable); // Register existing buffer as page-locked (pinned) memory
    uint8_t *h_output;
    cudaMallocHost(&h_output, WIDTH * HEIGHT);

    // profiling
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocate GPU memory
    uchar4 *d_pixels_in;
    uint8_t *d_pixels_out;
    HANDLE_ERROR(cudaMalloc(&d_pixels_in, size));
    HANDLE_ERROR(cudaMalloc(&d_pixels_out, WIDTH * HEIGHT)); // 1 byte per pixel

    // CUDA streams
    const int rows_per_stream = HEIGHT / N_STREAMS;
    cudaStream_t streams[N_STREAMS];

    for(int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);

        int start_row = i * rows_per_stream;
        uchar4 *slice_in = d_pixels_in + start_row * WIDTH;
        uint8_t *slice_out = d_pixels_out + start_row * WIDTH;
        size_t slice_size = rows_per_stream * WIDTH * sizeof(uchar4);

        cudaMemcpyAsync(slice_in, (uchar4*) image->pixels + start_row * WIDTH, slice_size, cudaMemcpyHostToDevice, streams[i]);

        dim3 block(BLOCK_X, BLOCK_Y);
        // Works for every image resolution
        dim3 grid((WIDTH + BLOCK_X*PIXELS_PER_THREAD - 1) / (BLOCK_X * PIXELS_PER_THREAD), // 240
                  (rows_per_stream + BLOCK_Y - 1) / BLOCK_Y); // 135

        rgb_to_grayscale_parallel<<<grid, block, 0, streams[i]>>>(slice_in, slice_out, WIDTH, rows_per_stream); // Run the kernel
        checkCUDAError((char*) cudaGetLastError());

        cudaMemcpyAsync(h_output + start_row * WIDTH, slice_out, rows_per_stream * WIDTH, cudaMemcpyDeviceToHost, streams[i]);
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

    SDL_UnlockSurface(image);

    // CUDA cleanup
    for (int i = 0; i < N_STREAMS; i++)
        HANDLE_ERROR(cudaStreamDestroy(streams[i]));
    HANDLE_ERROR(cudaFree(d_pixels_in));
    HANDLE_ERROR(cudaFree(d_pixels_out));
    cudaHostUnregister(image->pixels); // Unregister page-locked (pinned) memory


    // ----- Show the modified image -----
     SDL_Surface *grayscale = SDL_CreateRGBSurfaceWithFormatFrom(
        (uint8_t*) h_output,
        WIDTH,
        HEIGHT,
        8,      // Bit depth per pixel
        WIDTH,  // pitch (number of byte per row)
        SDL_PIXELFORMAT_INDEX8
    );
    // Creazione di una palette lineare: 0 = nero, 255 = bianco
    SDL_Color palette[256];
    for (int i = 0; i < 256; i++) {
        palette[i].r = i;
        palette[i].g = i;
        palette[i].b = i;
        palette[i].a = 255;
    }
    SDL_SetPaletteColors(grayscale->format->palette, palette, 0, 256);
    SDL_Surface* grayscale32 = SDL_ConvertSurfaceFormat(grayscale, SDL_PIXELFORMAT_RGBA32, 0);

    SDL_Window *window = create_window("RGB to grayscale - Optimized parallel"); // Create the window (fullscreen)
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, grayscale);
    SDL_FreeSurface(image); // Surface no longer needed
    SDL_FreeSurface(grayscale); // Surface no longer needed
    SDL_RenderCopy(renderer, texture, NULL, NULL); // Copy the texture to the renderer
    SDL_RenderPresent(renderer);     // Update the screen with the rendered content
    SDL_Delay(SHOW_IMAGE_FOR); // Show for 3 seconds

    // ----- Cleanup -----
    SDL_DestroyWindow(window);
    cudaFree(h_output);
    IMG_Quit();
    SDL_Quit();

    return 0;
}