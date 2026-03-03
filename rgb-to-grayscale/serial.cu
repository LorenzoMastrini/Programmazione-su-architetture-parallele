#include "../utils.h"

#define SHOW_IMAGE_FOR 3000 // How much time to show the image (3000 ms = 3 seconds)



// Conversione in-place di un'immagine da RGB a scala di grigi
void rgb_to_grayscale(SDL_Surface *img) {
    // Blocca la superficie per accedere ai pixel
    SDL_LockSurface(img);

    Uint32 *pixels = (Uint32*) img->pixels;
    int pixel_per_row = img->pitch / 4; // numero di pixel (ciascuno Uint32, 1byte) per riga (pitch = #byte per riga)

    for(int i = 0; i < pixel_per_row * img->h; i++) {
        Uint32 p = pixels[i]; // accedo a ogni pixel

            Uint8 r = (p >> 16) & 0xFF;
            Uint8 g = (p >> 8) & 0xFF;
            Uint8 b = p & 0xFF;

            // Grayscale (formula based on non-linear human perception of color)
            Uint8 gray = 0.3 * r + 0.59 * g + 0.11 * b;
            pixels[i] = (gray << 16) | (gray << 8) | gray;
    }

    SDL_UnlockSurface(img);
}


int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");

    // profiling
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    rgb_to_grayscale(image);
    
    // profiling
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Kernel time: %.3f ms\n", time);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    
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

    SDL_Delay(SHOW_IMAGE_FOR); // Show the image for 3 seconds

    // Free everything
    SDL_FreeSurface(image);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}