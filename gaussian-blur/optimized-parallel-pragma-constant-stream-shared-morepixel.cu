#include "../utils.h"
#include <stdio.h>

#define KERNEL_SIZE 21
#define BLOCK_SIZE 16
#define HALF 10
#define SIGMA 5.0
#define STREAM_POOL_SIZE 4
#define N_CHUNKS 16
#define PIXELS_PER_THREAD 2

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

__global__ void gaussian_horizontal(uchar4 __restrict__ *input,
    uchar4 __restrict__ *output,
    int width, int height){

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int base_x = blockIdx.x * blockDim.x * PIXELS_PER_THREAD;
    int x = base_x + tx * PIXELS_PER_THREAD;
    int y = blockIdx.y * blockDim.y + ty;

    if (y >= height) return;

    __shared__ uchar4 tile[BLOCK_SIZE][BLOCK_SIZE * PIXELS_PER_THREAD + 2 * HALF];

    for (int i = 0; i < PIXELS_PER_THREAD; i++) {
        int gx = x + i;
        int clamped_x = min(max(gx, 0), width - 1);

        if (gx < width)
        tile[ty][tx * PIXELS_PER_THREAD + i + HALF] =
        input[y * width + clamped_x];
    }

    // Halo
    if (tx < HALF) {
        int left_x  = base_x + tx - HALF;
        int right_x = base_x + tx + blockDim.x * PIXELS_PER_THREAD;

        int clamped_left  = min(max(left_x, 0), width - 1);
        int clamped_right = min(max(right_x, 0), width - 1);

        tile[ty][tx] =
        input[y * width + clamped_left];

        tile[ty][tx + HALF + blockDim.x * PIXELS_PER_THREAD] =
        input[y * width + clamped_right];
    }

    __syncthreads();

    for (int p = 0; p < PIXELS_PER_THREAD; p++) {

        int out_x = x + p;
        if (out_x >= width) continue;

        float r = 0, g = 0, b = 0;

        #pragma unroll
        for (int i = 0; i < KERNEL_SIZE; i++) {
        uchar4 pixel = tile[ty][tx * PIXELS_PER_THREAD + p + i];
        float w = d_kernel_1d[i];

        r += pixel.x * w;
        g += pixel.y * w;
        b += pixel.z * w;
        }

        uchar4 out;
        out.x = (unsigned char)r;
        out.y = (unsigned char)g;
        out.z = (unsigned char)b;
        out.w = tile[ty][tx * PIXELS_PER_THREAD + p + HALF].w;

        output[y * width + out_x] = out;
    }
}

__global__ void gaussian_vertical(uchar4 __restrict__ *input,
    uchar4 __restrict__ *output,
    int width, int height){

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int base_y = blockIdx.y * blockDim.y * PIXELS_PER_THREAD;
    int y = base_y + ty * PIXELS_PER_THREAD;

    if (x >= width) return;

    __shared__ uchar4 tile[BLOCK_SIZE * PIXELS_PER_THREAD + 2 * HALF][BLOCK_SIZE];

    for (int i = 0; i < PIXELS_PER_THREAD; i++) {
        int gy = y + i;
        int clamped_y = min(max(gy, 0), height - 1);

        if (gy < height)
        tile[ty * PIXELS_PER_THREAD + i + HALF][tx] =
        input[clamped_y * width + x];
    }

    // Halo
    if (ty < HALF) {
        int top_y    = base_y + ty - HALF;
        int bottom_y = base_y + ty + blockDim.y * PIXELS_PER_THREAD;

        int clamped_top    = min(max(top_y, 0), height - 1);
        int clamped_bottom = min(max(bottom_y, 0), height - 1);

        tile[ty][tx] =
        input[clamped_top * width + x];

        tile[ty + HALF + blockDim.y * PIXELS_PER_THREAD][tx] =
        input[clamped_bottom * width + x];
    }

    __syncthreads();

    for (int p = 0; p < PIXELS_PER_THREAD; p++) {
        int out_y = y + p;
        if (out_y >= height) continue;

        float r = 0, g = 0, b = 0;

        #pragma unroll
        for (int i = 0; i < KERNEL_SIZE; i++) {
            uchar4 pixel = tile[ty * PIXELS_PER_THREAD + p + i][tx];
            float w = d_kernel_1d[i];

            r += pixel.x * w;
            g += pixel.y * w;
            b += pixel.z * w;
        }

        uchar4 out;
        out.x = (unsigned char)r;
        out.y = (unsigned char)g;
        out.z = (unsigned char)b;
        out.w = tile[ty * PIXELS_PER_THREAD + p + HALF][tx].w;

        output[out_y * width + x] = out;
    }
}


// Funzione di Gaussian blur per immagine RGB
SDL_Surface* gaussian_blur(SDL_Surface* src) {
    SDL_Surface* dst = SDL_CreateRGBSurface(0, src->w, src->h, 32,
                                            src->format->Rmask, src->format->Gmask,
                                            src->format->Bmask, src->format->Amask);
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
    uchar4 *d_input, *d_temp;

    HANDLE_ERROR(cudaHostRegister(src->pixels, bytes, cudaHostRegisterPortable));
    HANDLE_ERROR(cudaHostRegister(dst->pixels, bytes, cudaHostRegisterPortable));

    cudaEventRecord(start);
    // Alloca memoria GPU
    HANDLE_ERROR(cudaMalloc(&d_input, bytes));
    HANDLE_ERROR(cudaMalloc(&d_temp, bytes));

    init_gaussian_kernel_once();

    const int rows_per_chunk = (height + N_CHUNKS - 1) / N_CHUNKS;
    cudaStream_t streams[STREAM_POOL_SIZE];

    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 256 thread per blocco
    dim3 grid((width + BLOCK_SIZE -1)/BLOCK_SIZE, (height + BLOCK_SIZE -1)/BLOCK_SIZE);
    
   // Crea un pool fisso di stream
   for (int s = 0; s < STREAM_POOL_SIZE; s++) {
        cudaStreamCreate(&streams[s]);
   }

   // Per ogni chunk: riusa uno stream del pool
   for (int i = 0; i < N_CHUNKS; i++) {
        int stream_id = i % STREAM_POOL_SIZE;
        cudaStream_t stream = streams[stream_id];

        HANDLE_ERROR(cudaStreamSynchronize(stream));

        int start_row = i * rows_per_chunk;
        if (start_row >= height) continue;

        int this_rows = min(rows_per_chunk, height - start_row);
        if (this_rows <= 0) continue;

        size_t slice_size = static_cast<size_t>(this_rows) * width * sizeof(uchar4);

        uchar4 *d_slice_in  = d_input + static_cast<size_t>(start_row) * width;
        uchar4 *d_slice_tmp = d_temp  + static_cast<size_t>(start_row) * width;
        uchar4 *h_slice_in  = reinterpret_cast<uchar4*>(src->pixels) + static_cast<size_t>(start_row) * width;
        uchar4 *h_slice_out = reinterpret_cast<uchar4*>(dst->pixels) + static_cast<size_t>(start_row) * width;

        // Copia H -> D della slice
        HANDLE_ERROR(cudaMemcpyAsync(d_slice_in,h_slice_in, slice_size, cudaMemcpyHostToDevice, stream));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 256 thread per blocco
        dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (this_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Kernel di blur orizzontale e verticale sulla slice
        gaussian_horizontal<<<grid, block, 0, stream>>>(d_slice_in, d_slice_tmp, width, this_rows);
        gaussian_vertical<<<grid, block, 0, stream>>>(d_slice_tmp, d_slice_in, width, this_rows);

        // Copia D -> H della slice
        HANDLE_ERROR(cudaMemcpyAsync(h_slice_out, d_slice_in, slice_size, cudaMemcpyDeviceToHost, stream));
    }

    for (int s = 0; s < STREAM_POOL_SIZE; s++) {
        HANDLE_ERROR(cudaStreamSynchronize(streams[s]));
        HANDLE_ERROR(cudaStreamDestroy(streams[s]));
    }
 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Execution time: %.3f ms\n", ms);

    HANDLE_ERROR(cudaFree(d_input));
    HANDLE_ERROR(cudaFree(d_temp));

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

