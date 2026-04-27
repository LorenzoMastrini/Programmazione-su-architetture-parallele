#include "../utils.h"
#include <stdio.h>

#define KERNEL_SIZE 21
#define BLOCK_SIZE 16
#define HALF 10
#define SIGMA 5.0
#define STREAM_POOL_SIZE 4
#define N_CHUNKS 16

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
}

void init_gaussian_kernel_once() {
    static bool initialized = false;
    if (initialized) return;

    generate_gaussian_kernel();
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_1d, h_kernel_1d, KERNEL_SIZE * sizeof(float)));

    initialized = true;
}

__global__ void gaussian_horizontal(uchar4 __restrict__ *input, uchar4 __restrict__ *output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    __shared__ uchar4 tile[BLOCK_SIZE][BLOCK_SIZE + 2 * HALF];
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

    float r = 0, g = 0, b = 0;

    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        uchar4 pixel = tile[ty][tx + i];
        float w = d_kernel_1d[i];

        r += pixel.x * w;
        g += pixel.y * w;
        b += pixel.z * w;
    }

    uchar4 out;
    out.x = (unsigned char)r;
    out.y = (unsigned char)g;
    out.z = (unsigned char)b;
    out.w = tile[ty][tx + HALF].w;
    
    output[y*width+x] = out;
}

__global__ void gaussian_vertical(uchar4 __restrict__ *input, uchar4 __restrict__ *output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    __shared__ uchar4 tile[BLOCK_SIZE + 2 * HALF][BLOCK_SIZE];
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

    float r = 0, g = 0, b = 0;

    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        uchar4 pixel = tile[ty + i][tx];
        float w = d_kernel_1d[i];

        r += pixel.x * w;
        g += pixel.y * w;
        b += pixel.z * w;
    }

    uchar4 out;
    out.x = (unsigned char)r;
    out.y = (unsigned char)g;
    out.z = (unsigned char)b;
    out.w = tile[ty + HALF][tx].w;
    
    output[y*width+x] = out;
}



int main(int argc, char* argv[]) {
    init_SDL();
    cudaFree(0);

    SDL_Surface* image = load_img(IMG_PATH);
    SDL_Surface* blurred = SDL_CreateRGBSurface(0, image->w, image->h, 32,
                                                image->format->Rmask, image->format->Gmask,
                                                image->format->Bmask, image->format->Amask);
    if (!blurred) {
        fprintf(stderr, "Errore creazione superficie destinazione\n");
        SDL_FreeSurface(image);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_LockSurface(image);
    SDL_LockSurface(blurred);

    int width = image->w;
    int height = image->h;
    size_t num_pixels = width * height;
    size_t bytes = num_pixels * sizeof(uchar4);
    const int rows_per_chunk = (height + N_CHUNKS - 1) / N_CHUNKS;
    const int max_halo_rows = min(height, rows_per_chunk + 2 * HALF);
    const size_t stream_buffer_size = (size_t)max_halo_rows * width * sizeof(uchar4);
    cudaStream_t streams[STREAM_POOL_SIZE];
    uchar4* d_stream_in[STREAM_POOL_SIZE];
    uchar4* d_stream_tmp[STREAM_POOL_SIZE];

    init_gaussian_kernel_once();

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    CUDA_CHECK(cudaHostRegister(image->pixels, bytes, cudaHostRegisterPortable));
    CUDA_CHECK(cudaHostRegister(blurred->pixels, bytes, cudaHostRegisterPortable));

    // Create stream pool of fixed size
    for (int s = 0; s < STREAM_POOL_SIZE; s++) {
        cudaStreamCreate(&streams[s]);
        CUDA_CHECK(cudaMalloc(&d_stream_in[s], stream_buffer_size));
        CUDA_CHECK(cudaMalloc(&d_stream_tmp[s], stream_buffer_size));
    }

    for (int i = 0; i < N_CHUNKS; i++) {
        int stream_id = i % STREAM_POOL_SIZE;
        cudaStream_t stream = streams[stream_id];

        CUDA_CHECK(cudaStreamSynchronize(stream));

        int start_row = i * rows_per_chunk;
        if (start_row >= height) continue;

        int end_row = min(start_row + rows_per_chunk, height);
        int halo_start = max(0, start_row - HALF);
        int halo_end = min(height, end_row + HALF);
        int halo_rows = halo_end - halo_start;
        int valid_rows = end_row - start_row;

        size_t halo_size = (size_t)halo_rows * width * sizeof(uchar4);
        size_t valid_size = (size_t)valid_rows * width * sizeof(uchar4);

        uchar4 *d_slice_in = d_stream_in[stream_id];
        uchar4 *d_slice_tmp = d_stream_tmp[stream_id];
        uchar4 *h_slice_in = reinterpret_cast<uchar4*>(image->pixels) + (size_t)halo_start * width;
        uchar4 *h_slice_out = reinterpret_cast<uchar4*>(blurred->pixels) + (size_t)start_row * width;

        CUDA_CHECK(cudaMemcpyAsync(d_slice_in, h_slice_in, halo_size, cudaMemcpyHostToDevice, stream));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (halo_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        gaussian_horizontal<<<grid, block, 0, stream>>>(d_slice_in, d_slice_tmp, width, halo_rows);
        CUDA_CHECK(cudaGetLastError());
        gaussian_vertical<<<grid, block, 0, stream>>>(d_slice_tmp, d_slice_in, width, halo_rows);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaMemcpyAsync(
            h_slice_out,
            d_slice_in + (size_t)(start_row - halo_start) * width,
            valid_size,
            cudaMemcpyDeviceToHost,
            stream
        ));
    }

    for (int s = 0; s < STREAM_POOL_SIZE; s++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        CUDA_CHECK(cudaFree(d_stream_in[s]));
        CUDA_CHECK(cudaFree(d_stream_tmp[s]));
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }

    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Total CUDA time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));  
    // -------------------------------------------------- END CUDA --------------------------------------------------

    SDL_UnlockSurface(image);
    SDL_UnlockSurface(blurred);

    show_image(reinterpret_cast<const uchar4*>(image->pixels), width, height, "Immagine Originale");
    show_image(reinterpret_cast<const uchar4*>(blurred->pixels), width, height, "Immagine Sfocata");
    
    SDL_FreeSurface(blurred);
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
}