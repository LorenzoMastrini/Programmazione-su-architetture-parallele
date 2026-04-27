#include "../utils.h"
#include <stdio.h>

#define KERNEL_SIZE 21
#define BLOCK_SIZE 32
#define HALF 10
#define SIGMA 5.0

float *d_kernel_1d = nullptr;

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

void init_gaussian_kernel() {
    static bool initialized = false;
    if (initialized) return;

    generate_gaussian_kernel();
    if (d_kernel_1d == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_kernel_1d, KERNEL_SIZE * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_kernel_1d, h_kernel_1d, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    initialized = true;
}

__global__ void gaussian_horizontal(uchar4 __restrict__ *input, uchar4 __restrict__ *output, const float *kernel, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float r = 0, g = 0, b = 0;

    for (int i = 0; i < KERNEL_SIZE; i++) {
        int k = i - HALF;
        int px = min(max(x + k, 0), width - 1);

        uchar4 pixel = input[y * width + px];
        float w = kernel[i];

        r += pixel.x * w;
        g += pixel.y * w;
        b += pixel.z * w;
    }


    uchar4 out;
    out.x = (unsigned char)r;
    out.y = (unsigned char)g;
    out.z = (unsigned char)b;
    out.w = input[y * width + x].w;

    output[y*width+x] = out;
}

__global__ void gaussian_vertical(uchar4 __restrict__ *input, uchar4 __restrict__ *output, const float *kernel, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float r = 0, g = 0, b = 0;

    
    for (int i = 0; i < KERNEL_SIZE; i++) {
        int k = i - HALF;
        int py = min(max(y + k, 0), height - 1);

        uchar4 pixel = input[py * width + x];
        float w = kernel[i];

        r += pixel.x * w;
        g += pixel.y * w;
        b += pixel.z * w;
    }

    uchar4 out;
    out.x = (unsigned char)r;
    out.y = (unsigned char)g;
    out.z = (unsigned char)b;
    out.w = input[y * width + x].w;
    
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
    uchar4 *d_input, *d_temp;

    init_gaussian_kernel();

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, image->pixels, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gaussian_horizontal<<<grid, block>>>(d_input, d_temp, d_kernel_1d, width, height);
    gaussian_vertical<<<grid, block>>>(d_temp, d_input, d_kernel_1d, width, height);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(blurred->pixels, d_input, bytes, cudaMemcpyDeviceToHost));

    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Total CUDA time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));  
    // -------------------------------------------------- END CUDA --------------------------------------------------

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_temp));

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