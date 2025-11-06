#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include "../utils.h"



__global__ void rgb_to_grayscale_parallel(Uint32* dev_pixels, int dev_w, int dev_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dev_w || y >= dev_h) return;

    int idx = y * dev_w + x;

    Uint32 p = dev_pixels[idx];
    Uint8 r = (p >> 16) & 0xFF;
    Uint8 g = (p >> 8) & 0xFF;
    Uint8 b = p & 0xFF;
    Uint8 gray = 0.3*r + 0.59*g + 0.11*b;

    dev_pixels[idx] = (gray << 16) | (gray << 8) | gray;
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("../images/img.jpg");



    Uint32 *pixels = (Uint32*) image->pixels;
    int size = image->w * image->h * sizeof(Uint32);

    // Alloca memoria GPU
    Uint32 *d_pixels;
    HANDLE_ERROR(cudaMalloc(&d_pixels, size));
    HANDLE_ERROR(cudaMemcpy(d_pixels, pixels, size, cudaMemcpyHostToDevice));

    dim3 block(16, 16); // 256 thread per blocco
    dim3 grid((image->w + block.x - 1) / block.x,
              (image->h + block.y - 1) / block.y);
    // Lancia il kernel
    rgb_to_grayscale_parallel<<<grid, block>>>(d_pixels, image->w, image->h);
    cudaDeviceSynchronize();
    checkCUDAError("Errore nella chiamata al kernel 'rgb_to_grayscale_parallel()'");

    // Copia i risultati indietro
    HANDLE_ERROR(cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(d_pixels));




    // Create the window to show the image using all the screen
    SDL_DisplayMode dm;
    if (SDL_GetDesktopDisplayMode(0, &dm) != 0) {
        printf("Errore SDL_GetDesktopDisplayMode: %s\n", SDL_GetError());
        return 1;
    }

    // Create the window (fullscreen)
    SDL_Window *window = SDL_CreateWindow("Edge detection",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          dm.w, dm.h, 0);
    
    if (!window) {
        printf("Errore SDL_CreateWindow: %s\n", SDL_GetError());
        return 1;
    }

    // Ottieni la superficie della finestra (dove disegneremo)
    SDL_Surface *screen = SDL_GetWindowSurface(window);
    if (!screen) {
        printf("Errore SDL_GetWindowSurface: %s\n", SDL_GetError());
        return 1;
    }

    // Copia la superficie nella finestra (scalandone correttamente le dimensioni)
    SDL_BlitScaled(image, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);

    SDL_Delay(3000); // Mostra per 3 secondi

    // Free everything
    SDL_FreeSurface(image);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();
    return 0;
}