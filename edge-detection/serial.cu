#include <math.h>
#include "../utils.h"

#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)
#define KERNEL_SIZE 5

// Kernel gaussiano 5x5 pre-calcolato e normalizzato
double kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1.0/273,  4.0/273,  7.0/273,  4.0/273, 1.0/273},
    {4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273},
    {7.0/273, 26.0/273, 41.0/273, 26.0/273, 7.0/273},
    {4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273},
    {1.0/273,  4.0/273,  7.0/273,  4.0/273, 1.0/273}
};



// Funzioni per leggere pixel da SDL_Surface 32bit
Uint32 get_pixel(SDL_Surface *surface, int x, int y) {
    if (x < 0) x = 0;
    if (x >= surface->w) x = surface->w - 1;
    if (y < 0) y = 0;
    if (y >= surface->h) y = surface->h - 1;
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * 4;
    return *(Uint32 *)p;
}

// Funzioni per scrivere pixel in una SDL_Surface 32bit
void put_pixel(SDL_Surface *surface, int x, int y, Uint32 pixel) {
    if (x < 0 || x >= surface->w || y < 0 || y >= surface->h) return;
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * 4;
    *(Uint32 *)p = pixel;
}



// ---------- GAUSSIAN BLUR ----------
SDL_Surface* gaussian_blur(SDL_Surface* src) {
    SDL_Surface* dst = SDL_CreateRGBSurface(0, src->w, src->h, 32,
                                            src->format->Rmask, src->format->Gmask,
                                            src->format->Bmask, src->format->Amask);
    if (!dst) {
        fprintf(stderr, "Errore creazione superficie destinazione\n");
        return NULL;
    }

    SDL_LockSurface(src);
    SDL_LockSurface(dst);

    int half = KERNEL_SIZE / 2;
    for (int y = 0; y < src->h; y++) {
        for (int x = 0; x < src->w; x++) {
            // Se il kernel uscisse fuori, copia il pixel originale
            /*if (x < half || x >= src->w - half || y < half || y >= src->h - half) {
                Uint32 pixel = get_pixel(src, x, y);
                put_pixel(dst, x, y, pixel);
                continue;
            }*/
            double r_sum = 0, g_sum = 0, b_sum = 0;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int px = x + kx;
                    int py = y + ky;
                    Uint32 pixel = get_pixel(src, px, py);

                    Uint8 r, g, b;
                    SDL_GetRGB(pixel, src->format, &r, &g, &b);

                    double weight = kernel[ky + half][kx + half];
                    r_sum += r * weight;
                    g_sum += g * weight;
                    b_sum += b * weight;
                }
            }
            Uint8 r_blur = (Uint8)r_sum;
            Uint8 g_blur = (Uint8)g_sum;
            Uint8 b_blur = (Uint8)b_sum;

            Uint32 new_pixel = SDL_MapRGB(dst->format, r_blur, g_blur, b_blur);
            put_pixel(dst, x, y, new_pixel);
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