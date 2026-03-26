#include <math.h>
#include "../utils.h"

// We work with 8K images (7680x4320 pixels)
#define KERNEL_SIZE 21
#define HALF 10
#define SIGMA 5.0
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)



// ---------- RGB TO GRAYSCALE ----------
__global__ void rgb_to_grayscale_parallel(uchar4* input, unsigned char *output, int w, int h) {
    // Global coordinates of the pixel in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x; // Index of the current pixel in the linear array

    // This data type also contains alpha channel, but it is ignored
    uchar4 p = input[idx];
    float r = p.x; // Red
    float g = p.y; // Green
    float b = p.z; // Blue

    // Grayscale formula (based on the non-linear human perception of color)
    float gray = 0.3*r + 0.59*g + 0.11*b;

    output[idx] = (unsigned char) gray;
}



// ---------- GAUSSIAN BLUR ----------
__constant__ float d_kernel[KERNEL_SIZE*KERNEL_SIZE];

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

__global__ void gaussian_blur_parallel(unsigned char *input, unsigned char *output, int width, int height, int pixels_per_row){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    double sum = 0.0;

    for (int ky = -HALF; ky <= HALF; ky++) {
        for (int kx = -HALF; kx <= HALF; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int pixel_index = py * pixels_per_row + px;

            unsigned char p = input[pixel_index];

            double weight = d_kernel[(ky + HALF) * KERNEL_SIZE + (kx + HALF)];
            sum += p * weight;
        }
    }

    int out_index = y * pixels_per_row + x;

    // Clamping dei valori nel range [0, 255]
    int out = (int) sum;

    out = max(0, min(255, out));

    unsigned char gray = (unsigned char) out;

    output[out_index] = gray;
}



// ---------- SOBEL EDGE DETECTION ----------
__global__ void sobel_edge_detection_parallel(unsigned char *input, unsigned char *output, int width, int height) {
    // Global coordinates of the pixel in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Excludes borders and out-of-bound values
    if (x == 0 || x >= width - 1) return;
    if (y == 0 || y >= height - 1) return;

    int idx = y * width + x; // Index of the current pixel in the linear array
    
    // Sobel matrices
    // x
    const int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    // y
    const int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int sumX = 0;
    int sumY = 0;

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            unsigned char pixel = input[(y + j) * width + (x + i)];

            sumX += pixel * gx[j+1][i+1];
            sumY += pixel * gy[j+1][i+1];
        }
    }

    int magnitude = (int) sqrt((double) (sumX * sumX + sumY * sumY));
    if (magnitude > 255) magnitude = 255;
    if (magnitude < 0) magnitude = 0;

    output[idx] = (unsigned char) magnitude;
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");
    SDL_LockSurface(image); // Lock to access pixels
    const int width = image->w;
    const int height = image->h;
    int rgb_size = width * height * sizeof(uchar4); // Size of RGB images
    int grayscale_size = width * height * sizeof(unsigned char); // Size of grayscale images
    int pixels_per_row = image->pitch / sizeof(Uint32); // 1 pixel = 4 bytes, hence the division (pitch: bytes per row)

    // RGB-to-grayscale output (Host)
    unsigned char *h_output = (unsigned char*) malloc(grayscale_size);

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Create CUDA environment
    dim3 block(32, 32); // 32x32 = 1024 threads per block (max for 4080 SUPER)
    // Scales for every image resolution
    dim3 grid((width + block.x - 1) / block.x,
              (height+ block.y - 1) / block.y);


    // ---------- RGB TO GRAYSCALE ----------
    // Allocate GPU memory
    uchar4 *d_pixels;
    CUDA_CHECK(cudaMalloc(&d_pixels, rgb_size));
    CUDA_CHECK(cudaMemcpy(d_pixels, image->pixels, rgb_size, cudaMemcpyHostToDevice));
    unsigned char *d_grayscale;
    CUDA_CHECK(cudaMalloc(&d_grayscale, grayscale_size));

    // RGB to grayscale kernel
    rgb_to_grayscale_parallel<<<grid, block>>>(d_pixels, d_grayscale, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());


    // ---------- GAUSSIAN BLUR ----------
    unsigned char *d_blurred;
    CUDA_CHECK(cudaMalloc(&d_blurred, grayscale_size));
    
    // Genera e copia il kernel gaussiano nella memoria costante della GPU
    init_gaussian_kernel();
    // Kernel launch
    gaussian_blur_parallel<<<grid, block>>>(d_grayscale, d_blurred, width, height, pixels_per_row);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    
    // ---------- SOBEL EDGE DETECTION ----------
    unsigned char *d_edges;
    CUDA_CHECK(cudaMalloc(&d_edges, grayscale_size));

    // CUDA Kernel
    sobel_edge_detection_parallel<<<grid, block>>>(d_blurred, d_edges, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Copy back the results
    CUDA_CHECK(cudaMemcpy(h_output, d_edges, grayscale_size, cudaMemcpyDeviceToHost));    
    
    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Total CUDA time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // -------------------------------------------------- END CUDA --------------------------------------------------

    show_image(h_output, width, height, "Sobel Edge Detection - Versione parallela naive");

    // Free everything
    CUDA_CHECK(cudaFree(d_pixels));
    CUDA_CHECK(cudaFree(d_grayscale));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_edges));
    SDL_UnlockSurface(image);
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
}