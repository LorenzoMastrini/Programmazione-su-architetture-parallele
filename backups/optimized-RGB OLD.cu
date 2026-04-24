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

// Ogni thread elabora PIXELS_PER_THREAD pixel consecutivi sulla stessa riga,
// usando accessi vettoriali per massimizzare coalescing e banda di memoria.
__global__ void rgb_to_grayscale_parallel(const uchar4* __restrict__ input,
                                          unsigned char* __restrict__ output,
                                          int rows_per_stream,
                                          int vectors_per_row) {
    // Coordinate del thread all'interno della slice elaborata dallo stream
    const int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int thread_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Lavoriamo in "unità vettoriali": ogni unità corrisponde a PIXELS_PER_THREAD pixel
    const int vector_x = thread_x;

    // Outside horizontal/vertical limits of the slice
    if (vector_x >= vectors_per_row || thread_y >= rows_per_stream) return;

    // Indice del gruppo vettoriale (PIXELS_PER_THREAD pixel) in un array lineare
    const uint32_t vector_index = thread_y * vectors_per_row + vector_x;

    // Caricamento vettoriale: 4 pixel RGBA (16 byte) da memoria globale
    const uchar4* rgba_pixels = reinterpret_cast<const uchar4*>(input) + vector_index * PIXELS_PER_THREAD;

    // Buffer locale con i valori di luminanza (1 byte per pixel)
    unsigned char gray[PIXELS_PER_THREAD];

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

        gray[i] = static_cast<unsigned char>(y);
    }

    // Pack dei 4 byte di luminanza in un vettore uchar4
    // (funziona correttamente perché PIXELS_PER_THREAD è fissato a 4)
    uchar4 packed_gray = make_uchar4(
        gray[0],
        gray[1],
        gray[2],
        gray[3]
    );

    // Scrittura vettoriale: 4 valori di luminanza (4 byte) in un colpo solo
    reinterpret_cast<uchar4*>(output)[vector_index] = packed_gray;
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");
    SDL_LockSurface(image); // Lock to access pixels
    const int width = image->w;
    const int height = image->h;
    const int input_size = width * height * sizeof(uchar4);
    const int output_size = width * height * sizeof(unsigned char);

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Register existing buffer as page-locked (pinned) memory
    CUDA_CHECK(cudaHostRegister(image->pixels, input_size, cudaHostRegisterPortable));
    unsigned char *h_output; // Output image
    CUDA_CHECK(cudaMallocHost(&h_output, output_size));

    // Allocate GPU memory
    uchar4 *d_pixels_in;
    unsigned char *d_pixels_out;
    CUDA_CHECK(cudaMalloc(&d_pixels_in, input_size));
    CUDA_CHECK(cudaMalloc(&d_pixels_out, output_size)); // 1 byte per pixel

    // CUDA streams
    const int rows_per_stream = height / N_STREAMS;
    size_t slice_size = rows_per_stream * width * sizeof(uchar4);
    cudaStream_t streams[N_STREAMS];
    int vectors_per_row = width / PIXELS_PER_THREAD;

    for(int i = 0; i < N_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

        int start_row = i * rows_per_stream;
        uchar4 *slice_in = d_pixels_in + start_row * width;
        unsigned char *slice_out = d_pixels_out + start_row * width;

        CUDA_CHECK(cudaMemcpyAsync(slice_in,
                                   (uchar4*) image->pixels + start_row * width,
                                   slice_size,
                                   cudaMemcpyHostToDevice,
                                   streams[i]));

        dim3 block(BLOCK_X, BLOCK_Y);
        // Works for every image resolution
        dim3 grid((width + BLOCK_X*PIXELS_PER_THREAD - 1) / (BLOCK_X*PIXELS_PER_THREAD),
                  (rows_per_stream + BLOCK_Y - 1) / BLOCK_Y);

        // Run the kernel
        rgb_to_grayscale_parallel<<<grid, block, 0, streams[i]>>>(slice_in, slice_out, rows_per_stream, vectors_per_row);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_output + start_row * WIDTH,
                                   slice_out,
                                   rows_per_stream * width * sizeof(unsigned char),
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

    show_image(h_output, WIDTH, HEIGHT, "Conversione da RGB a scala di grigi - Versione parallela ottimizzata");

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