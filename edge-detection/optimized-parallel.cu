#include <math.h>
#include "../utils.h"

// We work with 8K images (7680x4320 pixels)
#define WIDTH 7680
#define HEIGHT 4320
// Block dimensions fine-tuned to optimize occupancy
#define BLOCK_X 32
#define BLOCK_Y 8

// RGB to grayscale constants
#define N_STREAMS 16
#define PIXELS_PER_THREAD 4

// Gaussian Blur constants
#define KERNEL_SIZE 21
#define HALF 10
#define SIGMA 5.0
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)





// ---------- RGB TO GRAYSCALE ----------
// Use constant memory to access constants to be used in the kernel
__constant__ __device__ float GRAY_COEF[3] = {0.2126f, 0.7152f, 0.0722f};
__constant__ __device__ uint32_t D_WIDTH = WIDTH;
__constant__ __device__ uint32_t VECTORS_PER_ROW = WIDTH / PIXELS_PER_THREAD; // assumiamo D_WIDTH multiplo di PIXELS_PER_THREAD

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





// ---------- GAUSSIAN BLUR ----------
__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];
float h_kernel[KERNEL_SIZE * KERNEL_SIZE];

// Genera il kernel gaussiano normalizzato a runtime
void generate_gaussian_kernel() {
    double sum = 0.0;
    
    for (int y = -HALF; y <= HALF; y++) {
        for (int x = -HALF; x <= HALF; x++) {
            double value = exp(-(x*x + y*y) / (2.0 * SIGMA * SIGMA));
            h_kernel[(y + HALF) * KERNEL_SIZE + (x + HALF)] = value;
            sum += value;
        }
    }
    
    // Normalizza
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        h_kernel[i] /= sum;
    }
}



void init_gaussian_kernel() {
    static bool done = false;
    if (done) return;

    generate_gaussian_kernel();
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    done = true;
}



__global__ void gaussian_blur_parallel(uint8_t *input, uint8_t *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // For our grayscale buffers, one byte per pixel and 'width' pixels per row
    int pitch = width;
    double sum = 0;

    for (int ky = -HALF; ky <= HALF; ky++) {
        for (int kx = -HALF; kx <= HALF; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int pixel_index = py * pitch + px;

            double weight = d_kernel[(ky + HALF) * KERNEL_SIZE + (kx + HALF)];
            sum += input[pixel_index] * weight;
        }
    }

    int out_index = y * pitch + x;
    output[out_index] = (uint8_t) sum;
}





// ---------- SOBEL EDGE DETECTION ----------
// Sobel operator on 8-bit grayscale images (1 byte per pixel)
__global__ void sobel_edge_detection_parallel(uint8_t __restrict__ *input,
                                              uint8_t __restrict__ *output,
                                              int width,
                                              int height,
                                              int rows_per_stream,
                                              int start_row) {
    // Shared memory tile (+2 to consider halo: borders of the image tile)
    __shared__ uint8_t tile[BLOCK_Y + 2][BLOCK_X + 2];

    // Thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= rows_per_stream) return;

    // Global coordinates in the image
    int global_y = start_row + y;

    // Coordinates in the tile (+1 cause we consider also the halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Center: each thread loads his pixel
    if (x < width && y < rows_per_stream)
        tile[ty][tx] = input[y * width + x];

    // Left halo (first column)
    if (threadIdx.x == 0 && x > 0)
        tile[ty][0] = input[y * width + (x - 1)];

    // Right halo (last column)
    if (threadIdx.x == blockDim.x - 1 && x < width - 1)
        tile[ty][blockDim.x + 1] = input[y * width + (x + 1)];

    // Upper halo (first row)
    if (threadIdx.y == 0) {
        if (y > 0) {
            // Riga precedente ancora all'interno della slice
            tile[0][tx] = input[(y - 1) * width + x];
        } else {
            // Non abbiamo accesso alla riga precedente (border dell'immagine o della slice):
            // approssimiamo replicando il valore centrale
            tile[0][tx] = tile[ty][tx];
        }
    }

    // Bottom halo (last row)
    if (threadIdx.y == blockDim.y - 1) {
        if (y + 1 < rows_per_stream && global_y < height - 1) {
            // Riga successiva ancora all'interno della slice
            tile[blockDim.y + 1][tx] = input[(y + 1) * width + x];
        } else {
            // Non abbiamo accesso alla riga successiva (border dell'immagine o della slice):
            // approssimiamo replicando il valore centrale
            tile[blockDim.y + 1][tx] = tile[ty][tx];
        }
    }

    // Halo corners
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (x > 0 && y > 0) {
            tile[0][0] = input[(y - 1) * width + (x - 1)];
        } else {
            tile[0][0] = tile[ty][tx];
        }
    }

    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        if (x < width - 1 && y > 0) {
            tile[0][blockDim.x + 1] = input[(y - 1) * width + (x + 1)];
        } else {
            tile[0][blockDim.x + 1] = tile[ty][tx];
        }
    }

    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        if (x > 0 && (y + 1) < rows_per_stream && global_y < height - 1) {
            tile[blockDim.y + 1][0] = input[(y + 1) * width + (x - 1)];
        } else {
            tile[blockDim.y + 1][0] = tile[ty][tx];
        }
    }

    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        if (x < width - 1 && (y + 1) < rows_per_stream && global_y < height - 1) {
            tile[blockDim.y + 1][blockDim.x + 1] = input[(y + 1) * width + (x + 1)];
        } else {
            tile[blockDim.y + 1][blockDim.x + 1] = tile[ty][tx];
        }
    }

    __syncthreads();

    // Sobel: insert values of Sobel matrices directly in the formula
    // Horizontal gradient
    int gx =
        -tile[ty-1][tx-1] + tile[ty-1][tx+1] +
        -2*tile[ty][tx-1] + 2*tile[ty][tx+1] +
        -tile[ty+1][tx-1] + tile[ty+1][tx+1];

    // Vertical gradient
    int gy =
        -tile[ty-1][tx-1] -2*tile[ty-1][tx] -tile[ty-1][tx+1] +
         tile[ty+1][tx-1] +2*tile[ty+1][tx] +tile[ty+1][tx+1];

    // Fast approximation of the real formula
    int magnitude = abs(gx) + abs(gy);
    // Clamping
    magnitude = min(magnitude, 255); // magnitude is always >= 0
    output[y * width + x] = (uint8_t) magnitude;
}





int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");
    SDL_LockSurface(image);

    int rgb_size = WIDTH * HEIGHT * sizeof(uchar4); // Size of RGB images (4 bytes per pixel, one for each channel)
    int grayscale_size = WIDTH * HEIGHT * sizeof(uint8_t); // Size of grayscale images (1 byte per pixel, only one channel)

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    
    // ---------- RGB TO GRAYSCALE ----------
    CUDA_CHECK(cudaHostRegister(image->pixels, rgb_size, cudaHostRegisterPortable)); // Register existing buffer as page-locked (pinned) memory
    uint8_t *grayscale;
    CUDA_CHECK(cudaMallocHost(&grayscale, grayscale_size));

    // Allocate GPU memory
    uchar4 *d_pixels_in;
    uint8_t *d_pixels_out;
    CUDA_CHECK(cudaMalloc(&d_pixels_in, rgb_size));
    CUDA_CHECK(cudaMalloc(&d_pixels_out, grayscale_size));

    // CUDA streams
    const int rows_per_stream = HEIGHT / N_STREAMS;
    size_t slice_in_size = rows_per_stream * WIDTH * sizeof(uchar4);
    size_t slice_out_size = rows_per_stream * WIDTH * sizeof(uint8_t);
    cudaStream_t streams[N_STREAMS];
    
    for(int i = 0; i < N_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

        int start_row = i * rows_per_stream;
        uchar4 *slice_in = d_pixels_in + start_row * WIDTH;
        uint8_t *slice_out = d_pixels_out + start_row * WIDTH;

        CUDA_CHECK(cudaMemcpyAsync(slice_in, (uchar4*) image->pixels + start_row * WIDTH, slice_in_size, cudaMemcpyHostToDevice, streams[i]));

        dim3 block_grayscale(BLOCK_X, BLOCK_Y);
        // Works for every image resolution
        dim3 grid_grayscale((WIDTH + BLOCK_X*PIXELS_PER_THREAD - 1) / (BLOCK_X * PIXELS_PER_THREAD), // 240
                            (rows_per_stream + BLOCK_Y - 1) / BLOCK_Y); // 135

        rgb_to_grayscale_parallel<<<grid_grayscale, block_grayscale, 0, streams[i]>>>(slice_in, slice_out, rows_per_stream); // Run the kernel
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(grayscale + start_row * WIDTH, slice_out, slice_out_size, cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all streams
    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));


    CUDA_CHECK(cudaHostUnregister(image->pixels));
    SDL_UnlockSurface(image);
    SDL_FreeSurface(image);


    // ---------- GAUSSIAN BLUR ----------
    // Create CUDA environment
    dim3 block_gaussian_blur(32, 32); // 32x32 = 1024 threads per block (max for 4080 SUPER)
    // Scales for every image resolution
    dim3 grid_gaussian_blur((WIDTH + block_gaussian_blur.x - 1) / block_gaussian_blur.x, // 240 in this case
                            (HEIGHT+ block_gaussian_blur.y - 1) / block_gaussian_blur.y); // 135 in this case

    SDL_Surface* blurred = SDL_CreateRGBSurface(0, WIDTH, HEIGHT, 8, // 8 bit per pixel (grayscale)
                                                0, 0, 0, 0); // Mask not needed for 8 bit surfaces
    if (!blurred) {
        fprintf(stderr, "Errore creazione superficie destinazione\n");
        return 1;
    }

    SDL_LockSurface(blurred);

    uint8_t *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, grayscale_size));
    CUDA_CHECK(cudaMalloc(&d_output, grayscale_size));

    CUDA_CHECK(cudaMemcpy(d_input, grayscale, grayscale_size, cudaMemcpyHostToDevice));
    
    // Genera e copia il kernel nella memoria costante della GPU
    init_gaussian_kernel();
    
    gaussian_blur_parallel<<<grid_gaussian_blur, block_gaussian_blur>>>(d_input, d_output, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(blurred->pixels, d_output, grayscale_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));


    // ---------- SOBEL EDGE DETECTION ----------
    // Allocate GPU memory
    uint8_t *d_input_sobel, *d_output_sobel;
    CUDA_CHECK(cudaMalloc(&d_input_sobel, grayscale_size));
    CUDA_CHECK(cudaMalloc(&d_output_sobel, grayscale_size));
    
    // Host output buffer (pinned memory)
    uint8_t *h_output_sobel;
    CUDA_CHECK(cudaMallocHost(&h_output_sobel, grayscale_size));

    // Register existing buffer as page-locked (pinned) memory
    CUDA_CHECK(cudaHostRegister(blurred->pixels, grayscale_size, cudaHostRegisterPortable));

    // CUDA streams (same image slicing used in RGB-to-grayscale algorithm
    size_t slice_size = rows_per_stream * WIDTH * sizeof(uint8_t);
    cudaStream_t streams_sobel[N_STREAMS];

    for(int i = 0; i < N_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams_sobel[i]));

        int start_row = i * rows_per_stream;
        uint8_t *slice_in = d_input_sobel + start_row * WIDTH;
        uint8_t *slice_out = d_output_sobel + start_row * WIDTH;

        CUDA_CHECK(cudaMemcpyAsync(slice_in,
                                   ((uint8_t*) blurred->pixels) + start_row * WIDTH,
                                   slice_size,
                                   cudaMemcpyHostToDevice,
                                   streams_sobel[i]));

        dim3 block_sobel(BLOCK_X, BLOCK_Y);
        // Works for every image resolution
        dim3 grid_sobel((WIDTH + block_sobel.x - 1) / block_sobel.x,
                        (rows_per_stream + block_sobel.y - 1) / block_sobel.y);

        // Run the kernel
        sobel_edge_detection_parallel<<<grid_sobel, block_sobel, 0, streams_sobel[i]>>>(slice_in,
                                                                                        slice_out,
                                                                                        WIDTH,
                                                                                        HEIGHT,
                                                                                        rows_per_stream,
                                                                                        start_row);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_output_sobel + start_row * WIDTH, //Destination
                                     slice_out, // Source
                                     rows_per_stream * WIDTH * sizeof(uint8_t), // Size
                                     cudaMemcpyDeviceToHost, // Copy direction
                                     streams_sobel[i] // Stream
                                    ));
    }

    // Wait for all streams
    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cudaStreamSynchronize(streams_sobel[i]));

    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // -------------------------------------------------- END CUDA --------------------------------------------------

    // Create an 8-bit indexed surface directly from the Sobel output
    SDL_Surface *final8 = SDL_CreateRGBSurfaceWithFormatFrom(
        (uint8_t*) h_output_sobel,
        WIDTH,
        HEIGHT,
        8,                          // Bits per pixel
        WIDTH * sizeof(uint8_t),    // Pitch (bytes per row)
        SDL_PIXELFORMAT_INDEX8
    );

    // Linear grayscale palette: 0 = black, 255 = white
    SDL_Color palette[256];
    for (int i = 0; i < 256; i++) {
        palette[i].r = i;
        palette[i].g = i;
        palette[i].b = i;
        palette[i].a = 255;
    }
    SDL_SetPaletteColors(final8->format->palette, palette, 0, 256);

    // Convert to a 32-bit RGBA surface for easy blitting to the window surface
    SDL_Surface *final = SDL_ConvertSurfaceFormat(final8, SDL_PIXELFORMAT_RGBA32, 0);
    
    SDL_Window *window = create_window("RGB to grayscale");
    if (!window) {
        fprintf(stderr, "Errore nella creazione della finestra: %s\n", SDL_GetError());
        SDL_FreeSurface(blurred);
        SDL_FreeSurface(final);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    // Get the surface window (where we will paint the image)
    SDL_Surface *screen = SDL_GetWindowSurface(window);
    if (!screen) {
        printf("Errore SDL_GetWindowSurface: %s\n", SDL_GetError());
        SDL_FreeSurface(final);
        SDL_DestroyWindow(window);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    // Copy the image in the window surface (correctly scaling the dimensions)
    SDL_BlitScaled(final, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);

    SDL_Delay(SHOW_IMAGE_FOR); // Show the image for 3 seconds

    // ----- Clean up -----
    CUDA_CHECK(cudaFree(d_input_sobel));
    CUDA_CHECK(cudaFree(d_output_sobel));
    CUDA_CHECK(cudaFreeHost(h_output_sobel));
    SDL_UnlockSurface(blurred);
    SDL_FreeSurface(blurred);
    SDL_FreeSurface(final);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}