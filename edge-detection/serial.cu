#include "../utils.h"

#define KERNEL_SIZE 21
#define HALF 10
#define SIGMA 5.0
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)



// Conversione di un'immagine da RGB a scala di grigi
void rgb_to_grayscale(uchar4 *pixels, unsigned char *output, int number_of_pixels) {
    for(int i = 0; i < number_of_pixels; i++) {
        // Access each pixel
        uchar4 p = pixels[i];

        float r = p.x; // Red
        float g = p.y; // Green
        float b = p.z; // Blue

        // Grayscale formula (based on non-linear human perception of color)
        unsigned char gray = 0.3 * r + 0.59 * g + 0.11 * b;
        output[i] = gray;
    }
}



// Host Gaussian Blur kernel
float h_kernel[KERNEL_SIZE * KERNEL_SIZE];

// Normalized gaussian kernel generation
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

// Gaussian blur for grayscale image
void gaussian_blur(unsigned char* src, unsigned char *dst, int width, int height, int pitch) {
    // Generate normalized gaussian kernel at runtime
    generate_gaussian_kernel();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            double sum = 0.0;

            for (int ky = -HALF; ky <= HALF; ky++) {
                for (int kx = -HALF; kx <= HALF; kx++) {

                    int px = x + kx;
                    int py = y + ky;

                    // Clamp borders
                    if (px < 0) px = 0;
                    if (py < 0) py = 0;
                    if (px >= width)  px = width - 1;
                    if (py >= height) py = height - 1;

                    int pixel_index = py * pitch + px;
                    double weight = h_kernel[(ky + HALF) * KERNEL_SIZE + (kx + HALF)];

                    sum += src[pixel_index] * weight;
                }
            }

            int out_index = y * pitch + x;
            dst[out_index] = (unsigned char) sum;
        }
    }
}



// ---------- SOBEL EDGE DETECTION ----------
void sobel_edge_detection(unsigned char *input, unsigned char *output, int width, int height, int pixels_per_row) {
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

    // Sobel filter (excluding borders)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0;
            int sumY = 0;

            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    unsigned char pixel = input[(y + j) * pixels_per_row + (x + i)];

                    sumX += pixel * gx[j+1][i+1];
                    sumY += pixel * gy[j+1][i+1];
                }
            }

            int magnitude = (int) sqrt((double) (sumX * sumX + sumY * sumY));
            if (magnitude > 255) magnitude = 255;
            if (magnitude < 0) magnitude = 0;

            output[y * pixels_per_row + x] = magnitude;
        }
    }
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img(IMG_PATH);
    SDL_LockSurface(image); // Lock to access pixels
    const int width = image->w;
    const int height = image->h;
    const int grayscale_pitch = image->pitch / 4;
    const int number_of_pixels = width * height;

    uchar4 *pixels = (uchar4*) image->pixels;
    
    // RGB to grayscale output
    unsigned char* grayscale = (unsigned char*) malloc(number_of_pixels * sizeof(unsigned char));
    // Gaussian Blur output
    unsigned char* blurred = (unsigned char*) malloc(number_of_pixels * sizeof(unsigned char));
    // Edge detection output
    unsigned char* edges = (unsigned char*) malloc(number_of_pixels * sizeof(unsigned char));

    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    
    // RGB to grayscale conversion
    rgb_to_grayscale(pixels, grayscale, number_of_pixels);

    // Gaussian Blur
    gaussian_blur(grayscale, blurred, width, height, grayscale_pitch);

    // Edge detection
    sobel_edge_detection(blurred, edges, width, height, grayscale_pitch);
    
    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    show_image(edges, width, height, "Sobel Edge Detection - Versione seriale");

    // ----- Clean up -----
    SDL_UnlockSurface(image);
    SDL_FreeSurface(image);
    free(grayscale);
    free(blurred);
    free(edges);
    IMG_Quit();
    SDL_Quit();

    return 0;
}