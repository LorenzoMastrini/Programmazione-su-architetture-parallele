#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include "../utils.h"



// Conversione in-place di un'immagine da RGB a scala di grigi
void rgb_to_grayscale(SDL_Surface *img) {
    // Blocca la superficie per accedere ai pixel
    SDL_LockSurface(img);

    Uint32 *pixels = (Uint32*) img->pixels;
    int pixel_per_row = img->pitch / 4; // numero di pixel (ciascuno Uint32) per riga (pitch = #byte per riga)

    for (int y = 0; y < img->h; y++) {
        for (int x = 0; x < img->w; x++) {
            Uint32 p = pixels[y * pixel_per_row + x]; // accedo a ogni pixel

            Uint8 r = (p >> 16) & 0xFF;
            Uint8 g = (p >> 8) & 0xFF;
            Uint8 b = p & 0xFF;

            // Scala di grigi
            Uint8 gray = 0.3 * r + 0.59 * g + 0.11 * b;
            pixels[y * pixel_per_row + x] = (gray << 16) | (gray << 8) | gray;
        }
    }

    SDL_UnlockSurface(img);
}


int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    rgb_to_grayscale(image);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    

    SDL_Window *window = create_window("RGB to grayscale");

    // Get the surface window (where we will paint the image)
    SDL_Surface *screen = SDL_GetWindowSurface(window);
    if (!screen) {
        printf("Errore SDL_GetWindowSurface: %s\n", SDL_GetError());
        return 1;
    }

    // Copy the image in the window surface (correctly scaling the dimensions)
    SDL_BlitScaled(image, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);

    SDL_Delay(3000); // Show the image for 3 seconds

    // Free everything
    SDL_FreeSurface(image);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    printf("Kernel time: %.3f ms\n", ms);
    return 0;
}