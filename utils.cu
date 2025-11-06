#include "utils.h"



void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}



void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "ERRORE CUDA: >%s<: %s. Eseguo EXIT.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}



// Initialize SDL modules
int init_SDL() {
    // Load the graphics module of the SDL library
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("Errore SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    // Supported images formats
    if (!(IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG))) {
        printf("Errore IMG_Init: %s\n", IMG_GetError());
        return 1;
    }

    return 0;
}



// Load an image in a SDL_Surface
SDL_Surface* load_img(const char* path) {
    SDL_Surface *img = IMG_Load(path);
    if (!img) {
        printf("Errore IMG_Load: %s\n", IMG_GetError());
        exit(1);
    }

    // Convert the image pixels format to unify it
    SDL_Surface *converted = SDL_ConvertSurfaceFormat(img, SDL_PIXELFORMAT_RGB888, 0); // Each channel is made of 8 bit (1 byte) => each pixel is 32 bits (4 byte)
    SDL_FreeSurface(img);
    return converted;
}



SDL_Window* create_window(const char* title) {
    // Create the window using all the screen
    SDL_DisplayMode dm;
    if (SDL_GetDesktopDisplayMode(0, &dm) != 0) {
        printf("Errore SDL_GetDesktopDisplayMode: %s\n", SDL_GetError());
        exit(1);
    }

    SDL_Window *window = SDL_CreateWindow(title,
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          dm.w, dm.h, 0);
    if (!window) {
        printf("Errore SDL_CreateWindow: %s\n", SDL_GetError());
        exit(1);
    }

    return window;
}