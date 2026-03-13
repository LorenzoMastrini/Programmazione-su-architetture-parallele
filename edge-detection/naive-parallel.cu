#include <math.h>
#include "../utils.h"

// We work with 8K images (7680x4320 pixels)
#define WIDTH 7680
#define HEIGHT 4320
#define KERNEL_SIZE 21
#define HALF 10
#define SIGMA 5.0
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)



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

__global__ void gaussian_blur_parallel(Uint32 *input, Uint32 *output, int width, int height, int pixels_per_row){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    double r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;

    for (int ky = -HALF; ky <= HALF; ky++) {
        for (int kx = -HALF; kx <= HALF; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int pixel_index = py * pixels_per_row + px;

            Uint32 p = input[pixel_index];
            Uint8 r = (p >> 16) & 0xFF;
            Uint8 g = (p >> 8) & 0xFF;
            Uint8 b = p & 0xFF;

            double weight = d_kernel[(ky + HALF) * KERNEL_SIZE + (kx + HALF)];
            r_sum += r * weight;
            g_sum += g * weight;
            b_sum += b * weight;
        }
    }

    int out_index = y * pixels_per_row + x;

    // Clamping dei valori nel range [0, 255]
    int r_out = (int)r_sum;
    int g_out = (int)g_sum;
    int b_out = (int)b_sum;

    r_out = max(0, min(255, r_out));
    g_out = max(0, min(255, g_out));
    b_out = max(0, min(255, b_out));

    Uint8 r = (Uint8)r_out;
    Uint8 g = (Uint8)g_out;
    Uint8 b = (Uint8)b_out;

    Uint32 in_pixel = input[out_index];
    Uint8 a = (in_pixel >> 24) & 0xFF;

    Uint32 out_pixel =
        (a << 24) |
        (r << 16) |
        (g << 8)  |
        b;

    output[out_index] = out_pixel;
}



// ---------- RGB TO GRAYSCALE ----------
__global__ void rgb_to_grayscale_parallel(Uint32 *dev_pixels, int dev_w, int dev_h) {
    // Global coordinates of the pixel in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dev_w || y >= dev_h) return;

    int idx = y * dev_w + x; // Index of the current pixel in the linear array

    // This data type also contains alpha channel, but it is ignored
    Uint32 p = dev_pixels[idx];
    Uint8 r = (p >> 16) & 0xFF; // Red channel
    Uint8 g = (p >> 8) & 0xFF; // Green channel
    Uint8 b = p & 0xFF; // Blue channel
    Uint8 gray = 0.3*r + 0.59*g + 0.11*b; // Formula based on the non-linear human perception of color

    dev_pixels[idx] = (gray << 16) | (gray << 8) | gray;
}



// ---------- SOBEL EDGE DETECTION ----------
__global__ void sobel_edge_detection_parallel(Uint32 *input, Uint32 *output, int dev_w, int dev_h) {
    // Global coordinates of the pixel in the image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Excludes borders and out-of-bound values
    if (x == 0 || x >= dev_w - 1) return;
    if (y == 0 || y >= dev_h - 1) return;

    int idx = y * dev_w + x; // Index of the current pixel in the linear array
    
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
            Uint32 p = input[(y + j) * dev_w + (x + i)];

            // All channels are the same (gray value), anyone is valid
            Uint8 r = (p >> 16) & 0xFF; // Red channel
            Uint8 pixel = r;

            sumX += pixel * gx[j+1][i+1];
            sumY += pixel * gy[j+1][i+1];
        }
    }

    int magnitude = (int) sqrt((double) (sumX * sumX + sumY * sumY));
    if (magnitude > 255) magnitude = 255;
    if (magnitude < 0) magnitude = 0;

    Uint8 mag = (Uint8) magnitude;
    Uint32 new_pixel =
        (255 << 24) |   // A
        (mag << 16) |   // R
        (mag << 8)  |   // G
        mag;            // B
    output[idx] = new_pixel;
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");
    int size = WIDTH * HEIGHT * sizeof(Uint32); // Size of RGB images
    int pixels_per_row = image->pitch / sizeof(Uint32); // 1 pixel = 4 bytes, hence the division (pitch: bytes per row)

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
    dim3 grid((WIDTH + block.x - 1) / block.x, // 240 in this case
              (HEIGHT+ block.y - 1) / block.y); // 135 in this case


    // ---------- GAUSSIAN BLUR ----------
    SDL_Surface* blurred = SDL_CreateRGBSurface(0, image->w, image->h, 32,
                                                image->format->Rmask, image->format->Gmask,
                                                image->format->Bmask, image->format->Amask);
    if (!blurred) {
        fprintf(stderr, "Errore creazione superficie destinazione\n");
        return 1;
    }

    SDL_LockSurface(image);
    SDL_LockSurface(blurred);

    Uint32 *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(d_input, image->pixels, size, cudaMemcpyHostToDevice));
    
    // Genera e copia il kernel nella memoria costante della GPU
    init_gaussian_kernel();
    
    gaussian_blur_parallel<<<grid, block>>>(d_input, d_output, WIDTH, HEIGHT, pixels_per_row);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(blurred->pixels, d_output, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    
    // ---------- RGB TO GRAYSCALE ----------
    // Allocate GPU memory
    Uint32 *d_pixels;
    CUDA_CHECK(cudaMalloc(&d_pixels, size));
    CUDA_CHECK(cudaMemcpy(d_pixels, blurred->pixels, size, cudaMemcpyHostToDevice));

    // CUDA Kernel
    rgb_to_grayscale_parallel<<<grid, block>>>(d_pixels, WIDTH, HEIGHT);
    CUDA_CHECK(cudaGetLastError());

    // Copy back the results
    CUDA_CHECK(cudaMemcpy(blurred->pixels, d_pixels, size, cudaMemcpyDeviceToHost)); // Implicit barrier
    CUDA_CHECK(cudaFree(d_pixels));

    
    // ---------- SOBEL EDGE DETECTION ----------
    // Create CUDA environment
    dim3 block_sobel(32, 32); // 32x32 = 1024 threads per block (max for 4080 SUPER)
    // Scales for every image resolution
    dim3 grid_sobel((WIDTH + block_sobel.x - 1) / block_sobel.x, // 240 in this case
                    (HEIGHT+ block_sobel.y - 1) / block_sobel.y); // 135 in this case

    Uint32 *d_input_sobel, *d_output_sobel;
    CUDA_CHECK(cudaMalloc(&d_input_sobel, size));
    CUDA_CHECK(cudaMalloc(&d_output_sobel, size));
    CUDA_CHECK(cudaMemcpy(d_input_sobel, blurred->pixels, size, cudaMemcpyHostToDevice));

    // CUDA Kernel
    sobel_edge_detection_parallel<<<grid_sobel, block_sobel>>>(d_input_sobel, d_output_sobel, WIDTH, HEIGHT);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    // Host output buffer
    Uint32 *h_output = (Uint32*) malloc(size);

    // Copy back the results
    CUDA_CHECK(cudaMemcpy(h_output, d_output_sobel, size, cudaMemcpyDeviceToHost));    
    
    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // -------------------------------------------------- END CUDA --------------------------------------------------

    // Create SDL surfcce (ARGB8888)
    SDL_Surface *final = SDL_CreateRGBSurfaceFrom(
        (void*) h_output,          // Pixel data
        WIDTH,
        HEIGHT,
        32,                       // Depth
        WIDTH * sizeof(Uint32),   // Pitch
        0x00FF0000,               // Rmask
        0x0000FF00,               // Gmask
        0x000000FF,               // Bmask
        0xFF000000                // Amask
    );
    
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

    // Free everything
    CUDA_CHECK(cudaFree(d_input_sobel));
    CUDA_CHECK(cudaFree(d_output_sobel));
    SDL_UnlockSurface(image);
    SDL_FreeSurface(image);
    SDL_UnlockSurface(blurred);
    SDL_FreeSurface(blurred);
    SDL_FreeSurface(final);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}