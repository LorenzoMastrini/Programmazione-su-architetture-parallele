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
    HANDLE_ERROR(cudaMemcpyToSymbol(d_kernel_1d, h_kernel_1d, KERNEL_SIZE * sizeof(float)));

    initialized = true;
}

__global__ void gaussian_horizontal(uchar4 __restrict__ *input, uchar4 __restrict__ *output, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    __shared__ uchar4 tile[BLOCK_SIZE][BLOCK_SIZE + 2 * HALF];
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


// Funzione di Gaussian blur per immagine RGB
SDL_Surface* gaussian_blur(SDL_Surface* src) {
    SDL_Surface* dst = SDL_CreateRGBSurface(0, src->w, src->h, 32, src->format->Rmask, src->format->Gmask, src->format->Bmask, src->format->Amask);
    if (!dst) {
        fprintf(stderr, "Errore creazione superficie destinazione\n");
        return NULL;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    SDL_LockSurface(src);
    SDL_LockSurface(dst);

    int width = src->w;
    int height = src->h;

    size_t num_pixels = width * height;
    size_t bytes = num_pixels * sizeof(uchar4);

    HANDLE_ERROR(cudaHostRegister(src->pixels, bytes, cudaHostRegisterPortable));
    HANDLE_ERROR(cudaHostRegister(dst->pixels, bytes, cudaHostRegisterPortable));

    cudaEventRecord(start);
    init_gaussian_kernel_once();

    const int rows_per_chunk = (height + N_CHUNKS - 1) / N_CHUNKS;
    const int max_halo_rows = min(height, rows_per_chunk + 2 * HALF);
    const size_t stream_buffer_size = (size_t)max_halo_rows * width * sizeof(uchar4);
    cudaStream_t streams[STREAM_POOL_SIZE];
    uchar4* d_stream_in[STREAM_POOL_SIZE];
    uchar4* d_stream_tmp[STREAM_POOL_SIZE];

    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 256 thread per blocco
    dim3 grid((width + BLOCK_SIZE -1)/BLOCK_SIZE, (height + BLOCK_SIZE -1)/BLOCK_SIZE);
    
   // Crea un pool fisso di stream e relativi buffer privati.
   for (int s = 0; s < STREAM_POOL_SIZE; s++) {
        cudaStreamCreate(&streams[s]);
        HANDLE_ERROR(cudaMalloc(&d_stream_in[s], stream_buffer_size));
        HANDLE_ERROR(cudaMalloc(&d_stream_tmp[s], stream_buffer_size));
   }

   
   for (int i = 0; i < N_CHUNKS; i++) {
        int stream_id = i % STREAM_POOL_SIZE;
        cudaStream_t stream = streams[stream_id];

        HANDLE_ERROR(cudaStreamSynchronize(stream));

        int start_row = i * rows_per_chunk;
        if (start_row >= height) continue;

        int end_row = min(start_row + rows_per_chunk, height);

        
        int halo_start = max(0, start_row - HALF);
        int halo_end   = min(height, end_row + HALF);

        int halo_rows  = halo_end - halo_start;
        int valid_rows = end_row - start_row;

        // Dimensione con halo
        size_t halo_size  = (size_t)halo_rows * width * sizeof(uchar4);
        size_t valid_size = (size_t)valid_rows * width * sizeof(uchar4);

        // Buffer device privati dello stream corrente
        uchar4 *d_slice_in  = d_stream_in[stream_id];
        uchar4 *d_slice_tmp = d_stream_tmp[stream_id];

        uchar4 *h_slice_in  = reinterpret_cast<uchar4*>(src->pixels) + (size_t)halo_start * width;
        uchar4 *h_slice_out = reinterpret_cast<uchar4*>(dst->pixels) + (size_t)start_row * width;

        
        HANDLE_ERROR(cudaMemcpyAsync(d_slice_in, h_slice_in, halo_size, cudaMemcpyHostToDevice, stream));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (halo_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        
        gaussian_horizontal<<<grid, block, 0, stream>>>(d_slice_in, d_slice_tmp, width, halo_rows);

        gaussian_vertical<<<grid, block, 0, stream>>>(d_slice_tmp, d_slice_in, width, halo_rows);

        HANDLE_ERROR(cudaMemcpyAsync(h_slice_out, d_slice_in + (size_t)(start_row - halo_start) * width, valid_size, cudaMemcpyDeviceToHost, stream));
    }   

    for(int s = 0; s < STREAM_POOL_SIZE; s++) {
        HANDLE_ERROR(cudaStreamSynchronize(streams[s]));
        HANDLE_ERROR(cudaFree(d_stream_in[s]));
        HANDLE_ERROR(cudaFree(d_stream_tmp[s]));
        HANDLE_ERROR(cudaStreamDestroy(streams[s]));
    }
 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Execution time: %.3f ms\n", ms);

    SDL_UnlockSurface(src);
    SDL_UnlockSurface(dst);

    return dst;
}

int main(int argc, char* argv[]) {
    init_SDL();
    cudaFree(0);
    
    SDL_DisplayMode dm;
    if (SDL_GetDesktopDisplayMode(0, &dm) != 0) {
        fprintf(stderr, "Errore SDL_GetDesktopDisplayMode: %s\n", SDL_GetError());
        IMG_Quit();
        SDL_Quit();
        return 1;
    }
    
    SDL_Window *windowNormal = create_window("Immagine Originale");
    SDL_Window *windowBlur = create_window("Immagine Sfocata");
    
    if (!windowNormal || !windowBlur) {
        fprintf(stderr, "Impossibile creare le finestre: %s\n", SDL_GetError());
        if (windowNormal) SDL_DestroyWindow(windowNormal);
        if (windowBlur) SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }
    
    SDL_SetWindowPosition(windowNormal, 0, 0);
    SDL_SetWindowPosition(windowBlur, dm.w / 2, 0);
    
    SDL_Surface *screenNormal = SDL_GetWindowSurface(windowNormal);
    if (!screenNormal) {
        fprintf(stderr, "Impossibile ottenere la surface della finestra: %s\n", SDL_GetError());
        SDL_DestroyWindow(windowNormal);
        SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_Surface *screenBlur = SDL_GetWindowSurface(windowBlur);
    if (!screenBlur) {
        fprintf(stderr, "Impossibile ottenere la surface della finestra: %s\n", SDL_GetError());
        SDL_DestroyWindow(windowNormal);
        SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_Surface* image = load_img("images/img.jpg");
    
    SDL_Surface* blurred = gaussian_blur(image);

    if (!blurred) {
        fprintf(stderr, "Errore durante il Gaussian blur\n");
        SDL_FreeSurface(image);
        SDL_DestroyWindow(windowNormal);
        SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;

    }

    SDL_SaveBMP(blurred, "immagine_sfocata.bmp");

    SDL_BlitScaled(image, NULL, screenNormal, NULL);
    SDL_UpdateWindowSurface(windowNormal);
    
    SDL_BlitScaled(blurred, NULL, screenBlur, NULL);
    SDL_UpdateWindowSurface(windowBlur);
    
    SDL_Delay(4000);
    
    SDL_DestroyWindow(windowNormal);
    SDL_DestroyWindow(windowBlur);
    
    SDL_FreeSurface(blurred);
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();


    return 0;
}

