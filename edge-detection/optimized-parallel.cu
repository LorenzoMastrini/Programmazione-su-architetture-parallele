#include <math.h>
#include "../utils.h"

// Block dimensions fine-tuned to optimize occupancy
#define BLOCK_X 32
#define BLOCK_Y 8

// RGB to grayscale constants
#define N_STREAMS 8
#define NUM_SLICES 8 // Image height must be divisible by NUM_SLICES
#define PIXELS_PER_THREAD 4

// Gaussian Blur constants
#define KERNEL_SIZE 21
#define BLOCK_SIZE 32
#define HALF 10
#define SIGMA 5.0
#define STREAM_POOL_SIZE 4
#define N_CHUNKS 16

#define HALO_TOTAL (HALF + 1) // Gaussian (10) + Sobel (1) = 11




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

__global__ void gaussian_horizontal(unsigned char *input, unsigned char *output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    __shared__ unsigned char tile[BLOCK_SIZE][BLOCK_SIZE + 2 * HALF];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_x = blockIdx.x * blockDim.x;
    int center_x = base_x + tx;
    int left_x = base_x + tx - HALF;
    int right_x = base_x + tx + blockDim.x;
    int clamped_center_x = min(max(center_x, 0), width - 1);

    tile[ty][tx + HALF] = input[y * width + clamped_center_x];

    if (tx < HALF) {
        int clamped_left_x = min(max(left_x, 0), width - 1);
        int clamped_right_x = min(max(right_x, 0), width - 1);
        tile[ty][tx] = input[y * width + clamped_left_x];
        tile[ty][tx + HALF + blockDim.x] = input[y * width + clamped_right_x];
    }

    __syncthreads();

    float gray = 0;

    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        unsigned char pixel = tile[ty][tx + i];
        float w = d_kernel_1d[i];

        gray += pixel * w;
    }
    
    output[y*width+x] = (unsigned char) gray;
}

__global__ void gaussian_vertical(unsigned char *input, unsigned char *output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    __shared__ unsigned char tile[BLOCK_SIZE + 2 * HALF][BLOCK_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_y = blockIdx.y * blockDim.y;
    int center_y = base_y + ty;
    int top_y = base_y + ty - HALF;
    int bottom_y = base_y + ty + blockDim.y;
    int clamped_center_y = min(max(center_y, 0), height - 1);

    tile[ty + HALF][tx] = input[clamped_center_y * width + x];

    if (ty < HALF) {
        int clamped_top_y = min(max(top_y, 0), height - 1);
        int clamped_bottom_y = min(max(bottom_y, 0), height - 1);
        tile[ty][tx] = input[clamped_top_y * width + x];
        tile[ty + HALF + blockDim.y][tx] = input[clamped_bottom_y * width + x];
    }

    __syncthreads();

    float gray = 0;

    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        unsigned char pixel = tile[ty + i][tx];
        float w = d_kernel_1d[i];

        gray += pixel * w;
    }
    
    output[y*width+x] = (unsigned char) gray;
}



// ---------- SOBEL EDGE DETECTION ----------
/*
__global__ void sobel_edge_detection_parallel(unsigned char *input,
                                              unsigned char *output,
                                              int slice_width,
                                              int slice_height) {
    extern __shared__ unsigned char tile[];

    const int shared_width  = blockDim.x + 2;
    const int shared_height = blockDim.y + 2;

    // Thread id lineare nel blocco
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;

    // 🔹 Caricamento cooperativo del tile (centro + halo)
    int total = shared_width * shared_height;

    for (int i = tid; i < total; i += num_threads) {
        int local_y = i / shared_width;
        int local_x = i % shared_width;

        int global_x = blockIdx.x * blockDim.x + local_x - 1;
        int global_y = blockIdx.y * blockDim.y + local_y - 1;

        // Clamp ai bordi
        global_x = max(0, min(slice_width  - 1, global_x));
        global_y = max(0, min(slice_height - 1, global_y));

        tile[local_y * shared_width + local_x] =
            input[global_y * slice_width + global_x];
    }

    __syncthreads();

    // 🔹 Coordinate globali del pixel da calcolare
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= slice_width || y >= slice_height) return;

    // Coordinate nel tile (shift di +1 per halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // 🔹 Sobel
    int gx =
        -tile[(ty - 1) * shared_width + (tx - 1)] +
         tile[(ty - 1) * shared_width + (tx + 1)] +
        -2 * tile[ty * shared_width + (tx - 1)] +
         2 * tile[ty * shared_width + (tx + 1)] +
        -tile[(ty + 1) * shared_width + (tx - 1)] +
         tile[(ty + 1) * shared_width + (tx + 1)];

    int gy =
        -tile[(ty - 1) * shared_width + (tx - 1)] -
         2 * tile[(ty - 1) * shared_width + tx] -
         tile[(ty - 1) * shared_width + (tx + 1)] +
         tile[(ty + 1) * shared_width + (tx - 1)] +
         2 * tile[(ty + 1) * shared_width + tx] +
         tile[(ty + 1) * shared_width + (tx + 1)];

    int magnitude = abs(gx) + abs(gy);
    magnitude = min(magnitude, 255);

    output[y * slice_width + x] = (unsigned char)magnitude;
} // End sobel_edge_detection_parallel()
*/

// Sobel operator on 8-bit grayscale images (1 byte per pixel)
__global__ void sobel_edge_detection_parallel(unsigned char *input,
                                              unsigned char *output,
                                              int slice_width,
                                              int slice_height) {
    // Shared memory 2D (same size as block)
    __shared__ unsigned char tile[BLOCK_Y + 2][BLOCK_X + 2];

    // Coordinate nel tile
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    // Coordinate globali
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Centro
    int cx = max(0, min(slice_width - 1, x));
    int cy = max(0, min(slice_height - 1, y));
    tile[ty][tx] = input[cy * slice_width + cx];

    // Halo sinistro
    if (threadIdx.x == 0) {
        int hx = max(0, min(slice_width - 1, x - 1));
        int hy = max(0, min(slice_height - 1, y));
        tile[ty][0] = input[hy * slice_width + hx];
    }

    // Halo destro
    if (threadIdx.x == blockDim.x - 1) {
        int hx = max(0, min(slice_width - 1, x + 1));
        int hy = max(0, min(slice_height - 1, y));
        tile[ty][BLOCK_X + 1] = input[hy * slice_width + hx];
    }

    // Halo superiore
    if (threadIdx.y == 0) {
        int hx = max(0, min(slice_width - 1, x));
        int hy = max(0, min(slice_height - 1, y - 1));
        tile[0][tx] = input[hy * slice_width + hx];
    }

    // Halo inferiore
    if (threadIdx.y == blockDim.y - 1) {
        int hx = max(0, min(slice_width - 1, x));
        int hy = max(0, min(slice_height - 1, y + 1));
        tile[BLOCK_Y + 1][tx] = input[hy * slice_width + hx];
    }

    // Angoli
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int hx = max(0, min(slice_width - 1, x - 1));
        int hy = max(0, min(slice_height - 1, y - 1));
        tile[0][0] = input[hy * slice_width + hx];
    }

    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        int hx = max(0, min(slice_width - 1, x + 1));
        int hy = max(0, min(slice_height - 1, y - 1));
        tile[0][BLOCK_X + 1] = input[hy * slice_width + hx];
    }

    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        int hx = max(0, min(slice_width - 1, x - 1));
        int hy = max(0, min(slice_height - 1, y + 1));
        tile[BLOCK_Y + 1][0] = input[hy * slice_width + hx];
    }

    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        int hx = max(0, min(slice_width - 1, x + 1));
        int hy = max(0, min(slice_height - 1, y + 1));
        tile[BLOCK_Y + 1][BLOCK_X + 1] = input[hy * slice_width + hx];
    }

    __syncthreads();

    // Sobel
    if (x < slice_width && y < slice_height) {
        int gx = -tile[ty - 1][tx - 1] + tile[ty - 1][tx + 1]
               - 2 * tile[ty][tx - 1] + 2 * tile[ty][tx + 1]
               - tile[ty + 1][tx - 1] + tile[ty + 1][tx + 1];

        int gy = -tile[ty - 1][tx - 1] - 2 * tile[ty - 1][tx] - tile[ty - 1][tx + 1]
               + tile[ty + 1][tx - 1] + 2 * tile[ty + 1][tx] + tile[ty + 1][tx + 1];

        int magnitude = abs(gx) + abs(gy);
        magnitude = min(magnitude, 255);

        output[y * slice_width + x] = (unsigned char)magnitude;
    }
} // End sobel_edge_detection_parallel()



int main() {
    init_SDL();

    SDL_Surface *image = load_img(IMG_PATH);
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
    unsigned char *d_work; // Working array in all the pipeline
    CUDA_CHECK(cudaMalloc(&d_input, rgb_size));
    CUDA_CHECK(cudaMalloc(&d_work, grayscale_size));

    // CUDA streams
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

    // Number of rows in each slide of the the image, each handled by a stream
    const int rows_per_slice = height / NUM_SLICES;    

    // Get the number of Streaming Multiprocessor of the GPU
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

    dim3 block(256);
    dim3 grid(32 * numSMs); // Optimize occupacy

    unsigned char *d_temp_gaussian;
    CUDA_CHECK(cudaMalloc(&d_temp_gaussian, grayscale_size));

    init_gaussian_kernel_once();

    dim3 block_gaussian(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_gaussian(
        (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (rows_per_slice + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // Grid and block for Sobel (must match BLOCK_X and BLOCK_Y)
    dim3 block_sobel(BLOCK_X, BLOCK_Y);
    dim3 grid_sobel(
        (width + BLOCK_X - 1) / BLOCK_X,
        (rows_per_slice + BLOCK_Y - 1) / BLOCK_Y
    );

    // Stream cycle
    for (int i = 0; i < NUM_SLICES; i++) {
        int s = i % N_STREAMS;

        int start_row = i * rows_per_slice;
        int end_row   = start_row + rows_per_slice;

        // --- Estensione con halo ---
        int copy_start = max(0, start_row - HALO_TOTAL);
        int copy_end   = min(height, end_row + HALO_TOTAL);

        int extended_rows = copy_end - copy_start;

        // Offset interno (dove inizia la parte "valida")
        int local_offset = start_row - copy_start;

        // Dimensioni memoria
        size_t in_size  = extended_rows * width * sizeof(uchar4);
        size_t out_size = rows_per_slice * width * sizeof(unsigned char);

        // Puntatori device
        uchar4 *slice_input  = d_input  + copy_start * width;
        unsigned char *slice_work = d_work + copy_start * width;
        unsigned char *slice_tmp  = d_temp_gaussian + copy_start * width;

        // ---------------- HtoD ----------------
        CUDA_CHECK(cudaMemcpyAsync(
            slice_input,
            (uchar4*)image->pixels + copy_start * width,
            in_size,
            cudaMemcpyHostToDevice,
            streams[s]
        ));

        // ---------------- RGB → GRAY ----------------
        rgb_to_grayscale_parallel<<<grid, block, 0, streams[s]>>>(
            slice_input,
            slice_work,
            width,
            extended_rows
        );
        CUDA_CHECK(cudaGetLastError());

        // ---------------- GAUSSIAN ----------------
        dim3 grid_gaussian_local(
            (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (extended_rows + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

        gaussian_horizontal<<<grid_gaussian_local, block_gaussian, 0, streams[s]>>>(
            slice_work,
            slice_tmp,
            width,
            extended_rows
        );
        CUDA_CHECK(cudaGetLastError());

        gaussian_vertical<<<grid_gaussian_local, block_gaussian, 0, streams[s]>>>(
            slice_tmp,
            slice_work,
            width,
            extended_rows
        );
        CUDA_CHECK(cudaGetLastError());

        // ---------------- SOBEL ----------------
        dim3 grid_sobel_local(
            (width + BLOCK_X - 1) / BLOCK_X,
            (extended_rows + BLOCK_Y - 1) / BLOCK_Y
        );

        sobel_edge_detection_parallel<<<grid_sobel_local, block_sobel, 0, streams[s]>>>(
            slice_work,
            slice_tmp,
            width,
            extended_rows
        );
        CUDA_CHECK(cudaGetLastError());

        // ---------------- DtoH (solo parte valida) ----------------
        CUDA_CHECK(cudaMemcpyAsync(
            h_output + start_row * width,
            slice_tmp + local_offset * width,
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
    SDL_free(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
} // End main()