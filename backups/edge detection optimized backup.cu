#include <math.h>
#include "../utils.h"

// Block dimensions fine-tuned to optimize occupancy
#define BLOCK_X 16
#define BLOCK_Y 16

// RGB to grayscale constants
#define N_STREAMS 8
#define NUM_SLICES 8 // Image height must be divisible by NUM_SLICES

// Gaussian Blur constants
#define KERNEL_SIZE 21
#define HALF 10
#define SIGMA 5.0



// ---------- RGB TO GRAYSCALE ----------
// Use constant memory to access constants to be used in the kernel
__constant__ float GRAY_COEF[3] = {0.213f, 0.715f, 0.072f};

__device__ __forceinline__ unsigned char to_gray(const uchar4 p) {
    // fmaf(a, b, c): (a x b) + c
    return (unsigned char) fmaf(GRAY_COEF[2], p.z, fmaf(GRAY_COEF[1], p.y, GRAY_COEF[0] * p.x));
} // End to-gray()



// ---------- GAUSSIAN BLUR ----------
__constant__ float d_kernel_1d[KERNEL_SIZE];
float h_kernel_1d[KERNEL_SIZE];

void generate_gaussian_kernel() {
    double sum = 0.0;
    for (int i = -HALF; i <= HALF; i++) {
        double v = exp(-(i*i) / (2.0 * SIGMA * SIGMA));
        h_kernel_1d[i + HALF] = v;
        sum += v;
    }

    for (int i = 0; i < KERNEL_SIZE; i++)
        h_kernel_1d[i] /= sum;
} // End generate_gaussian_kernel()

void init_gaussian_kernel_once() {
    static bool initialized = false;
    if (initialized) return;

    generate_gaussian_kernel();
    HANDLE_ERROR(cudaMemcpyToSymbol(d_kernel_1d, h_kernel_1d, KERNEL_SIZE * sizeof(float)));

    initialized = true;
} //End init_gaussian_kernel_once()



// Sobel operator on 8-bit grayscale images (1 byte per pixel)
__global__ void full_edge_detection_parallel(uchar4 *input,
                                             unsigned char *output,
                                             int slice_width,
                                             int slice_height) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;

    if (x >= slice_width || y >= slice_height) return;
    
    // TILE con halo per Gaussian (21x21)
    __shared__ float tile[BLOCK_Y + 2*HALF][BLOCK_X + 2*HALF];

    // =============================
    // LOAD TILE + HALO (GLOBAL → SHARED)
    // =============================
    for (int dy = ty; dy < BLOCK_Y + 2*HALF; dy += blockDim.y) {
        for (int dx = tx; dx < BLOCK_X + 2*HALF; dx += blockDim.x) {

            int gx = blockIdx.x * blockDim.x + dx - HALF;
            int gy = blockIdx.y * blockDim.y + dy - HALF;

            gx = min(max(gx, 0), slice_width - 1);
            gy = min(max(gy, 0), slice_height - 1);

            uchar4 p = input[gy * slice_width + gx];

            tile[dy][dx] = to_gray(p);
        }
    }

    __syncthreads();

    // =============================
    // GAUSSIAN BLUR (SEPARABILE)
    // =============================

    float blur_h = 0.f;

    #pragma unroll
    for (int k = -HALF; k <= HALF; k++) {
        blur_h += tile[ty + HALF][tx + HALF + k] * d_kernel_1d[k + HALF];
    }

    __shared__ float tile_blur_h[BLOCK_Y][BLOCK_X];
    tile_blur_h[ty][tx] = blur_h;

    __syncthreads();

    float blur = 0.f;

    #pragma unroll
    for (int k = -HALF; k <= HALF; k++) {
        int yy = min(max(ty + k, 0), BLOCK_Y - 1);
        blur += tile_blur_h[yy][tx] * d_kernel_1d[k + HALF];
    }

    __shared__ float tile_blur[BLOCK_Y + 2][BLOCK_X + 2];

    tile_blur[ty+1][tx+1] = blur;

    // =============================
    // LOAD SOBEL HALO
    // =============================
    if (tx == 0)
        tile_blur[ty+1][0] = tile_blur[ty+1][1];

    if (tx == BLOCK_X-1)
        tile_blur[ty+1][BLOCK_X+1] = tile_blur[ty+1][BLOCK_X];

    if (ty == 0)
        tile_blur[0][tx+1] = tile_blur[1][tx+1];

    if (ty == BLOCK_Y-1)
        tile_blur[BLOCK_Y+1][tx+1] = tile_blur[BLOCK_Y][tx+1];

    // angoli
    if (tx==0 && ty==0) tile_blur[0][0] = tile_blur[1][1];
    if (tx==BLOCK_X-1 && ty==0) tile_blur[0][BLOCK_X+1] = tile_blur[1][BLOCK_X];
    if (tx==0 && ty==BLOCK_Y-1) tile_blur[BLOCK_Y+1][0] = tile_blur[BLOCK_Y][1];
    if (tx==BLOCK_X-1 && ty==BLOCK_Y-1)
        tile_blur[BLOCK_Y+1][BLOCK_X+1] = tile_blur[BLOCK_Y][BLOCK_X];

    __syncthreads();

    // =============================
    // SOBEL
    // =============================
    float gx =
        -tile_blur[ty][tx]     + tile_blur[ty][tx+2]
        -2*tile_blur[ty+1][tx] + 2*tile_blur[ty+1][tx+2]
        -tile_blur[ty+2][tx]   + tile_blur[ty+2][tx+2];

    float gy =
        -tile_blur[ty][tx]     -2*tile_blur[ty][tx+1] - tile_blur[ty][tx+2]
        +tile_blur[ty+2][tx]   +2*tile_blur[ty+2][tx+1] + tile_blur[ty+2][tx+2];

    int mag = min((int)(fabsf(gx) + fabsf(gy)), 255);

    output[y * slice_width + x] = (unsigned char) mag;
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");
    SDL_LockSurface(image); // Lock to access pixels
    const int width = image->w;
    const int height = image->h;
    int rgb_size = width * height * sizeof(uchar4); // Size of RGB image (4 bytes per pixel)
    int grayscale_size = width * height * sizeof(unsigned char); // Size of grayscale images (1 byte per pixel)

    // Register existing buffer as page-locked (pinned) memory
    CUDA_CHECK(cudaHostRegister(image->pixels, rgb_size, cudaHostRegisterPortable));
    // Host output buffer (pinned memory)
    unsigned char *h_output;
    CUDA_CHECK(cudaMallocHost(&h_output, grayscale_size));

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Allocate GPU memory
    uchar4 *d_input; // Original RGB pixels
    unsigned char *d_output; // Working array in all the pipeline
    CUDA_CHECK(cudaMalloc(&d_input, rgb_size));
    CUDA_CHECK(cudaMalloc(&d_output, grayscale_size));

    // CUDA streams
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Number of rows in each slide of the the image, each handled by a stream
    const int rows_per_slice = height / NUM_SLICES;    

    init_gaussian_kernel_once();

    // Grid and block for Sobel (must match BLOCK_X and BLOCK_Y)
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (width + BLOCK_X - 1) / BLOCK_X,
        (rows_per_slice + BLOCK_Y - 1) / BLOCK_Y
    );

    // Stream cycle
    for (int i = 0; i < NUM_SLICES; i++) {
        int s = i % N_STREAMS;
        int start_row = i * rows_per_slice;

        size_t in_size  = rows_per_slice * width * sizeof(uchar4);
        size_t out_size  = rows_per_slice * width * sizeof(unsigned char);
        
        uchar4 *slice_input  = d_input  + start_row * width;
        unsigned char *slice_output = d_output + start_row * width;

        // HtoD
        CUDA_CHECK(cudaMemcpyAsync(slice_input,
                                   (uchar4*)image->pixels + start_row * width,
                                   in_size,
                                   cudaMemcpyHostToDevice,
                                   streams[s]));

        full_edge_detection_parallel<<<grid, block, 0, streams[s]>>>(
            slice_input,
            slice_output,
            width,
            rows_per_slice);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            h_output + start_row * width,
            slice_output,
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
    printf("Time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // -------------------------------------------------- END CUDA --------------------------------------------------

    show_image(h_output, width, height, "Sobel Edge Detection - Versione parallela ottimizzata");

    // ----- Clean up -----
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    //CUDA_CHECK(cudaFree(d_output_sobel));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaHostUnregister(image->pixels));
    SDL_free(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
} // End main()