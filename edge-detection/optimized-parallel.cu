#include <math.h>
#include "../utils.h"

// Block dimensions fine-tuned to optimize occupancy
#define BLOCK_X 32
#define BLOCK_Y 16

// RGB to grayscale constants
#define N_STREAMS 8
#define NUM_SLICES 8 // Image height must be divisible by NUM_SLICES
#define PIXELS_PER_THREAD 4

// Gaussian Blur constants
#define KERNEL_SIZE 21
#define HALF 10
#define SIGMA 5.0
// This is the number of rows to be added to the upper and lower bounds of each
// slice to avoid artifacts during Gaussian Blur and Sobel
#define HALO_ROWS HALF



// ---------- RGB TO GRAYSCALE ----------
// Use constant memory to access constants to be used in the kernel
__constant__ float GRAY_COEF[3] = {0.213f, 0.715f, 0.072f};

__device__ __forceinline__ unsigned char to_gray(const uchar4 p) {
    // fmaf(a, b, c): (a x b) + c
    return (unsigned char) fmaf(GRAY_COEF[2], p.z, fmaf(GRAY_COEF[1], p.y, GRAY_COEF[0] * p.x));
} // End to-gray()

// Ogni thread elabora PIXELS_PER_THREAD pixel consecutivi in memoria
// Il `grid-stride loop` è su un unico indice lineare, così `grid` e `block` possono essere 1D.
__global__ void rgb_to_grayscale_parallel(const uchar4* __restrict__ input,
                                          unsigned char* __restrict__ output,
                                          int slice_width,
                                          int slice_height) {
    const int total_pixels = slice_width * slice_height;

    // Grid-stride loop su indice lineare, ma distribuendo i PIXELS_PER_THREAD
    // con passo = blockDim.x per evitare overlap tra thread.
    const int base0 = blockIdx.x * blockDim.x * PIXELS_PER_THREAD + threadIdx.x;
    const int stride = gridDim.x * blockDim.x * PIXELS_PER_THREAD;

    for (int base = base0; base < total_pixels; base += stride) {
        #pragma unroll
        for (int i = 0; i < PIXELS_PER_THREAD; i++) {
            const int pixel_idx = base + i * blockDim.x;
            if (pixel_idx < total_pixels)
                output[pixel_idx] = to_gray(input[pixel_idx]);
        }
    }
} // End rgb_to_grayscale_parallel()



// ---------- GAUSSIAN BLUR ----------
__constant__ float d_kernel_1d[KERNEL_SIZE];

void generate_gaussian_kernel() {
    float h_kernel_1d[KERNEL_SIZE];

    float sum = 0.0;
    for (int i = -HALF; i <= HALF; i++) {
        float v = exp(-(i*i) / (2.0 * SIGMA * SIGMA));
        h_kernel_1d[i + HALF] = v;
        sum += v;
    }

    for (int i = 0; i < KERNEL_SIZE; i++)
        h_kernel_1d[i] /= sum;

    HANDLE_ERROR(cudaMemcpyToSymbol(d_kernel_1d, h_kernel_1d, KERNEL_SIZE * sizeof(float)));
} // End generate_gaussian_kernel()


__global__ void gaussian_horizontal(unsigned char *input, unsigned char *output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    __shared__ unsigned char tile[BLOCK_Y][BLOCK_X + 2 * HALF];
    // Coordinates in the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Base global coordinates of the block 
    int base_x = blockIdx.x * blockDim.x;
    
    // Loading of the center
    int gx = min(max(base_x + tx, 0), width - 1);
    tile[ty][tx + HALF] = input[y * width + gx];

    // Loading of the halo
    if (tx < HALF) {
        int gx_left = min(max(base_x + tx - HALF, 0), width - 1);
        int gx_right = min(max(base_x + tx + blockDim.x, 0), width - 1);
        
        tile[ty][tx] = input[y * width + gx_left];
        tile[ty][tx + HALF + blockDim.x] = input[y * width + gx_right];
    }

    __syncthreads();

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        unsigned char pixel = tile[ty][tx + i];
        float w = d_kernel_1d[i];

        sum += pixel * w;
    }
    
    output[y*width+x] = (unsigned char) sum;
}


__global__ void gaussian_vertical(unsigned char *input, unsigned char *output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    __shared__ unsigned char tile[BLOCK_Y + 2 * HALF][BLOCK_X];
    // Coordinats in the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_y = blockIdx.y * blockDim.y;

    // Loading of the center
    int gy = min(max(base_y + ty, 0), height - 1);
    tile[ty + HALF][tx] = input[gy * width + x];

    // Halo
    if (ty < HALF) {
        int gy_top = min(max(base_y + ty - HALF, 0), height - 1);
        int gy_bottom = min(max(base_y + ty + blockDim.y, 0), height - 1);
        tile[ty][tx] = input[gy_top * width + x];
        tile[ty + HALF + blockDim.y][tx] = input[gy_bottom * width + x];
    }

    __syncthreads();

    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        unsigned char pixel = tile[ty + i][tx];
        float w = d_kernel_1d[i];

        sum += pixel * w;
    }
    
    output[y*width+x] = (unsigned char) sum;
}



// ---------- SOBEL EDGE DETECTION ----------
// Sobel operator on 8-bit grayscale images (1 byte per pixel)
__global__ void sobel_edge_detection_parallel(unsigned char *input,
                                              unsigned char *output,
                                              int slice_width,
                                              int slice_height) {
    // Global coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > slice_width || y > slice_height) return;

    // Shared memory 2D (+ 2 for the halo)
    __shared__ unsigned char tile[BLOCK_Y + 2][BLOCK_X + 2];

    // Coordinates in the tile
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Center
    int cx = max(0, min(slice_width - 1, x));
    int cy = max(0, min(slice_height - 1, y));
    tile[ty][tx] = input[cy * slice_width + cx];

    // Left
    if (tx == 1) {
        int hx = max(0, min(slice_width - 1, x - 1));
        int hy = max(0, min(slice_height - 1, y));
        tile[ty][0] = input[hy * slice_width + hx];
    }

    // Right
    if (tx == blockDim.x) {
        int hx = max(0, min(slice_width - 1, x + 1));
        int hy = max(0, min(slice_height - 1, y));
        tile[ty][BLOCK_X + 1] = input[hy * slice_width + hx];
    }

    // Top
    if (ty == 1) {
        int hx = max(0, min(slice_width - 1, x));
        int hy = max(0, min(slice_height - 1, y - 1));
        tile[0][tx] = input[hy * slice_width + hx];
    }

    // Bottom
    if (ty == blockDim.y) {
        int hx = max(0, min(slice_width - 1, x));
        int hy = max(0, min(slice_height - 1, y + 1));
        tile[BLOCK_Y + 1][tx] = input[hy * slice_width + hx];
    }

    // Upper-Left corner
    if (tx == 1 && ty == 1) {
        int hx = max(0, min(slice_width - 1, x - 1));
        int hy = max(0, min(slice_height - 1, y - 1));
        tile[0][0] = input[hy * slice_width + hx];
    }

    // Upper-Right corner
    if (tx == blockDim.x && ty == 1) {
        int hx = max(0, min(slice_width - 1, x + 1));
        int hy = max(0, min(slice_height - 1, y - 1));
        tile[0][BLOCK_X + 1] = input[hy * slice_width + hx];
    }

    // Bottom-Left corner
    if (tx == 1 && ty == blockDim.y) {
        int hx = max(0, min(slice_width - 1, x - 1));
        int hy = max(0, min(slice_height - 1, y + 1));
        tile[BLOCK_Y + 1][0] = input[hy * slice_width + hx];
    }

    // Bottom-Right corner
    if (tx == blockDim.x && ty == blockDim.y) {
        int hx = max(0, min(slice_width - 1, x + 1));
        int hy = max(0, min(slice_height - 1, y + 1));
        tile[BLOCK_Y + 1][BLOCK_X + 1] = input[hy * slice_width + hx];
    }

    __syncthreads();

    // Sobel
    int gx =  -tile[ty - 1][tx - 1] + tile[ty - 1][tx + 1]
            -2*tile[ty][tx - 1] + 2 * tile[ty][tx + 1]
              -tile[ty + 1][tx - 1] + tile[ty + 1][tx + 1];

    int gy = - tile[ty - 1][tx - 1] - 2 * tile[ty - 1][tx] - tile[ty - 1][tx + 1]
             + tile[ty + 1][tx - 1] + 2 * tile[ty + 1][tx] + tile[ty + 1][tx + 1];

    int magnitude = min(abs(gx) + abs(gy), 255);

    output[y * slice_width + x] = (unsigned char) magnitude;
} // End sobel_edge_detection_parallel()



int main() {
    init_SDL();

    SDL_Surface *image = load_img(IMG_PATH);
    SDL_LockSurface(image); // Lock to access pixels
    const int width = image->w;
    const int height = image->h;
    int rgb_size = width * height * sizeof(uchar4); // Size of RGB image (4 bytes per pixel)
    int grayscale_size = width * height * sizeof(unsigned char); // Size of grayscale images (1 byte per pixel)

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Register existing buffer as page-locked (pinned) memory
    CUDA_CHECK(cudaHostRegister(image->pixels, rgb_size, cudaHostRegisterPortable));
    // Host output buffer (pinned memory)
    unsigned char *h_output;
    CUDA_CHECK(cudaMallocHost(&h_output, grayscale_size));

    // Allocate GPU memory
    uchar4 *d_input; // Original RGB pixels
    unsigned char *d_work, *d_temp_gaussian; // Working array in all the pipeline
    CUDA_CHECK(cudaMalloc(&d_input, rgb_size));
    CUDA_CHECK(cudaMalloc(&d_work, grayscale_size));
    CUDA_CHECK(cudaMalloc(&d_temp_gaussian, grayscale_size));

    // Create CUDA streams
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Number of rows in a slice, each handled by a stream
    const int slice_height = height / NUM_SLICES;    

    generate_gaussian_kernel();

    // Block for Gaussian Blur and Sobel
    dim3 block(BLOCK_X, BLOCK_Y);

    // Stream cycle
    for (int i = 0; i < NUM_SLICES; i++) {
        int s = i % N_STREAMS;

        int start_row = i * slice_height; // First row of the current slice
        int end_row   = start_row + slice_height; // Last row (excluded)

        // Halo
        int copy_start = max(0, start_row - HALO_ROWS); // Extend up
        int copy_end   = min(height, end_row + HALO_ROWS); // Extend down
        int extended_rows = copy_end - copy_start; // Total rows to copy and process

        // Memory size
        size_t in_size  = extended_rows * width * sizeof(uchar4);
        size_t out_size = slice_height * width * sizeof(unsigned char);

        // Device pointers
        uchar4 *slice_input  = d_input  + copy_start * width;
        unsigned char *slice_work = d_work + copy_start * width;
        unsigned char *slice_tmp  = d_temp_gaussian + copy_start * width;

        dim3 grid_local(
            (width + BLOCK_X - 1) / BLOCK_X,
            (extended_rows + BLOCK_Y - 1) / BLOCK_Y
        );

        // HtoD
        CUDA_CHECK(cudaMemcpyAsync(
            slice_input,
            (uchar4*)image->pixels + copy_start * width,
            in_size,
            cudaMemcpyHostToDevice,
            streams[s]
        ));

        // ---------------- RGB → GRAY ----------------
        int total_pixels = width * extended_rows;
        dim3 grid_gray((total_pixels / PIXELS_PER_THREAD + 255) / 256);

        rgb_to_grayscale_parallel<<<grid_gray, block, 0, streams[s]>>>(
            slice_input,
            slice_work,
            width,
            extended_rows
        );
        CUDA_CHECK(cudaGetLastError());

        // ---------------- GAUSSIAN BLUR ----------------
        gaussian_horizontal<<<grid_local, block, 0, streams[s]>>>(
            slice_work,
            slice_tmp,
            width,
            extended_rows
        );
        CUDA_CHECK(cudaGetLastError());

        gaussian_vertical<<<grid_local, block, 0, streams[s]>>>(
            slice_tmp,
            slice_work,
            width,
            extended_rows
        );
        CUDA_CHECK(cudaGetLastError());

        // ---------------- SOBEL ----------------
        sobel_edge_detection_parallel<<<grid_local, block, 0, streams[s]>>>(
            slice_work,
            slice_tmp,
            width,
            extended_rows
        );
        CUDA_CHECK(cudaGetLastError());

        // ---------------- DtoH (solo parte valida) ----------------
        CUDA_CHECK(cudaMemcpyAsync(
            h_output + start_row * width,
            slice_tmp + HALO_ROWS * width,
            out_size,
            cudaMemcpyDeviceToHost,
            streams[s]
        ));
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
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_temp_gaussian));
    //CUDA_CHECK(cudaFree(d_output_sobel));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaHostUnregister(image->pixels));
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
} // End main()