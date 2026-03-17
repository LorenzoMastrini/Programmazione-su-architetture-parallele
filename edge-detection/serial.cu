#include <math.h>
#include "../utils.h"

#define KERNEL_SIZE 21
#define HALF 10
#define SIGMA 5.0
#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)

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

// Funzione di Gaussian blur per immagine RGB
SDL_Surface* gaussian_blur(SDL_Surface* src) {
    SDL_Surface* dst = SDL_CreateRGBSurface(
        0,
        src->w, src->h, 32,
        src->format->Rmask,
        src->format->Gmask,
        src->format->Bmask,
        src->format->Amask
    );

    if (!dst) {
        fprintf(stderr, "Errore creazione superficie destinazione\n");
        return NULL;
    }

    SDL_LockSurface(src);
    SDL_LockSurface(dst);

    int width = src->w;
    int height = src->h;
    int pitch = src->pitch;

    unsigned char* input  = (unsigned char*)src->pixels;
    unsigned char* output = (unsigned char*)dst->pixels;

    generate_gaussian_kernel();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            double r_sum = 0.0;
            double g_sum = 0.0;
            double b_sum = 0.0;

            for (int ky = -HALF; ky <= HALF; ky++) {
                for (int kx = -HALF; kx <= HALF; kx++) {

                    int px = x + kx;
                    int py = y + ky;

                    // Clamp bordi
                    if (px < 0) px = 0;
                    if (py < 0) py = 0;
                    if (px >= width)  px = width - 1;
                    if (py >= height) py = height - 1;

                    int pixel_index = py * pitch + px * 4;
                    double weight = h_kernel[(ky + HALF) * KERNEL_SIZE + (kx + HALF)];

                    r_sum += input[pixel_index + 0] * weight;
                    g_sum += input[pixel_index + 1] * weight;
                    b_sum += input[pixel_index + 2] * weight;
                }
            }

            int out_index = y * pitch + x * 4;
            output[out_index + 0] = (unsigned char)r_sum;
            output[out_index + 1] = (unsigned char)g_sum;
            output[out_index + 2] = (unsigned char)b_sum;
            output[out_index + 3] = input[out_index + 3]; // alpha invariato
        }
    }

    SDL_UnlockSurface(src);
    SDL_UnlockSurface(dst);

    return dst;
}



// ---------- RGB TO GRAYSCALE ----------
void rgb_to_grayscale(SDL_Surface *img) {
    // Blocca la superficie per accedere ai pixel
    SDL_LockSurface(img);

    Uint32 *pixels = (Uint32*) img->pixels;
    int pixel_per_row = img->pitch / 4; // numero di pixel (ciascuno Uint32, 1byte) per riga (pitch = #byte per riga)

    for(int i = 0; i < pixel_per_row * img->h; i++) {
        Uint32 p = pixels[i]; // accedo a ogni pixel

        Uint8 r,g,b;
        SDL_GetRGB(p, img->format, &r, &g, &b);

        // Grayscale (formula based on non-linear human perception of color)
        Uint8 gray = 0.3 * r + 0.59 * g + 0.11 * b;
        pixels[i] = (gray << 16) | (gray << 8) | gray;
    }

    SDL_UnlockSurface(img);
}



// ---------- SOBEL EDGE DETECTION ----------
SDL_Surface* sobel_edge_detection(SDL_Surface *img) {
    int width = img->w;
    int height = img->h;
    
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
    
    // Gaussian Blur
    generate_gaussian_kernel();
    SDL_Surface* blurred = gaussian_blur(img);
    if (!blurred) return NULL;

    // Grayscale conversion (still 32 bit)
    rgb_to_grayscale(blurred);

    SDL_Surface* edges = SDL_CreateRGBSurface(
        0, width, height, 32,
        img->format->Rmask,
        img->format->Gmask,
        img->format->Bmask,
        img->format->Amask
    );
    if (!edges) {
        SDL_FreeSurface(blurred);
        return NULL;
    }

    SDL_LockSurface(blurred);
    SDL_LockSurface(edges);

    Uint32 *gray_pixels  = (Uint32*) blurred->pixels;
    Uint32 *edges_pixels = (Uint32*) edges->pixels;

    int pixels_per_row = blurred->pitch / 4; // pixels per row (1 pixel = 1 byte)

    // Sobel filter (excluding borders)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0;
            int sumY = 0;

            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    Uint32 p = gray_pixels[(y + j) * pixels_per_row + (x + i)];

                    // Need all channels to use SDL_GetRGB(), but they are all the same (gray value)
                    Uint8 r, g, b;
                    SDL_GetRGB(p, blurred->format, &r, &g, &b);
                    Uint8 pixel = r; // Any channel is valid

                    sumX += pixel * gx[j+1][i+1];
                    sumY += pixel * gy[j+1][i+1];
                }
            }

            int magnitude = (int) sqrt((double) (sumX * sumX + sumY * sumY));
            if (magnitude > 255) magnitude = 255;
            if (magnitude < 0) magnitude = 0;

            Uint8 mag = (Uint8) magnitude;
            Uint32 new_pixel = SDL_MapRGB(edges->format, mag, mag, mag);
            edges_pixels[y * pixels_per_row + x] = new_pixel;
        }
    }

    SDL_UnlockSurface(edges);
    SDL_UnlockSurface(blurred);
    SDL_FreeSurface(blurred);

    return edges;
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");

    // Profiling
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Edge detection
    SDL_Surface *final = sobel_edge_detection(image);
    if (!final) {
        fprintf(stderr, "Errore nell'elaborazione Sobel\n");
        HANDLE_ERROR(cudaEventDestroy(start));
        HANDLE_ERROR(cudaEventDestroy(stop));
        SDL_FreeSurface(image);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }
    
    // Profiling
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel time: %.3f ms\n", time);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    
    SDL_Window *window = create_window("RGB to grayscale");
    if (!window) {
        fprintf(stderr, "Errore nella creazione della finestra: %s\n", SDL_GetError());
        SDL_FreeSurface(image);
        SDL_FreeSurface(final);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    // Get the surface window (where we will paint the image)
    SDL_Surface *screen = SDL_GetWindowSurface(window);
    if (!screen) {
        printf("Errore SDL_GetWindowSurface: %s\n", SDL_GetError());
        SDL_FreeSurface(image);
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
    SDL_FreeSurface(image);
    SDL_FreeSurface(final);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}