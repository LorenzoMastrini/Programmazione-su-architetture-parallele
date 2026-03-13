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
__constant__ __device__ float GRAY_COEF[3] = {0.2126f, 0.7152f, 0.0722f};
__constant__ __device__ uint32_t VECTORS_PER_ROW = WIDTH / PIXELS_PER_THREAD; // assumiamo WIDTH multiplo di PIXELS_PER_THREAD

// Ogni thread elabora PIXELS_PER_THREAD pixel consecutivi sulla stessa riga,
// usando accessi vettoriali (uint4) per massimizzare coalescing e banda di memoria.
__global__ void rgb_to_grayscale_parallel(const uchar4* __restrict__ input,
                                          uint8_t* __restrict__ output,
                                          int rows_per_stream) {
    // Coordinate del thread all'interno della slice elaborata dallo stream
    const uint32_t thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t thread_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Lavoriamo in "unità vettoriali": ogni unità corrisponde a PIXELS_PER_THREAD pixel
    const uint32_t vector_x = thread_x;

    // Fuori dai limiti orizzontali o verticali della slice -> il thread non elabora nulla
    if (vector_x >= VECTORS_PER_ROW ||
        thread_y >= static_cast<uint32_t>(rows_per_stream)) {
        return;
    }

    // Indice del gruppo vettoriale (PIXELS_PER_THREAD pixel) in un array lineare di uint4
    const uint32_t vector_index = thread_y * VECTORS_PER_ROW + vector_x;

    // Caricamento vettoriale: 4 pixel RGBA (16 byte) da memoria globale
    const uint4 rgba_vec = reinterpret_cast<const uint4*>(input)[vector_index];

    // Interpretiamo i 16 byte di rgba_vec come 4 struct uchar4 consecutivi
    const uchar4* rgba_pixels = reinterpret_cast<const uchar4*>(&rgba_vec);

    // Buffer locale con i valori di luminanza (1 byte per pixel)
    uint8_t gray[PIXELS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < PIXELS_PER_THREAD; ++i) {
        const uchar4 p = rgba_pixels[i];

        const float r = static_cast<float>(p.x);
        const float g = static_cast<float>(p.y);
        const float b = static_cast<float>(p.z);

        // Conversione RGB -> luminanza usando le costanti GRAY_COEF in memoria costante
        const float y = GRAY_COEF[0] * r +
                        GRAY_COEF[1] * g +
                        GRAY_COEF[2] * b;

        gray[i] = static_cast<uint8_t>(y);
    }

    // Pack dei 4 byte di luminanza in una singola word a 32 bit
    // (funziona correttamente perché PIXELS_PER_THREAD è fissato a 4)
    uint32_t packed_gray =
        (static_cast<uint32_t>(gray[0])      ) |
        (static_cast<uint32_t>(gray[1]) <<  8) |
        (static_cast<uint32_t>(gray[2]) << 16) |
        (static_cast<uint32_t>(gray[3]) << 24);

    // Scrittura vettoriale: 4 valori di luminanza (4 byte) in un colpo solo
    reinterpret_cast<uint32_t*>(output)[vector_index] = packed_gray;
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");
    SDL_LockSurface(image);
    int size = WIDTH * HEIGHT * sizeof(uchar4);

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Register existing buffer as page-locked (pinned) memory
    CUDA_CHECK(cudaHostRegister(image->pixels, size, cudaHostRegisterPortable));
    uint8_t *h_output;
    CUDA_CHECK(cudaMallocHost(&h_output, WIDTH * HEIGHT));

    // Allocate GPU memory
    uchar4 *d_pixels_in;
    uint8_t *d_pixels_out;
    CUDA_CHECK(cudaMalloc(&d_pixels_in, size));
    CUDA_CHECK(cudaMalloc(&d_pixels_out, WIDTH * HEIGHT * sizeof(uint8_t))); // 1 byte per pixel

    // CUDA streams
    const int rows_per_stream = HEIGHT / N_STREAMS;
    size_t slice_size = rows_per_stream * WIDTH * sizeof(uchar4);
    cudaStream_t streams[N_STREAMS];
    
    for(int i = 0; i < N_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

        int start_row = i * rows_per_stream;
        uchar4 *slice_in = d_pixels_in + start_row * WIDTH;
        uint8_t *slice_out = d_pixels_out + start_row * WIDTH;

        CUDA_CHECK(cudaMemcpyAsync(slice_in,
                                   (uchar4*) image->pixels + start_row * WIDTH,
                                   slice_size,
                                   cudaMemcpyHostToDevice,
                                   streams[i]));

        dim3 block(BLOCK_X, BLOCK_Y);
        // Works for every image resolution
        dim3 grid((WIDTH + BLOCK_X*PIXELS_PER_THREAD - 1) / (BLOCK_X * PIXELS_PER_THREAD), // 240
                  (rows_per_stream + BLOCK_Y - 1) / BLOCK_Y); // 135

        rgb_to_grayscale_parallel<<<grid, block, 0, streams[i]>>>(slice_in, slice_out, rows_per_stream); // Run the kernel
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_output + start_row * WIDTH,
                                   slice_out,
                                   rows_per_stream * WIDTH,
                                   cudaMemcpyDeviceToHost,
                                   streams[i]));
    }

    // Wait for all streams
    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));  

    // -------------------------------------------------- END CUDA --------------------------------------------------

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
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, grayscale32);
    SDL_RenderCopy(renderer, texture, NULL, NULL); // Copy the texture to the renderer
    SDL_RenderPresent(renderer); // Update the screen with the rendered content
    SDL_Delay(SHOW_IMAGE_FOR); // Show for 3 seconds

    // ----- Cleanup -----
    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    CUDA_CHECK(cudaFree(d_pixels_in));
    CUDA_CHECK(cudaFree(d_pixels_out));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaHostUnregister(image->pixels)); // Unregister page-locked (pinned) memory
    SDL_UnlockSurface(image);
    SDL_FreeSurface(image);
    SDL_FreeSurface(grayscale32);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}