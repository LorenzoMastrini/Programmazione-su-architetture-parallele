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
    SDL_Surface *converted = SDL_ConvertSurfaceFormat(img, SDL_PIXELFORMAT_RGBA32, 0); // each pixel is 32 bits (4 byte)
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



// Grayscale version
void show_image(const unsigned char* grayscale_image, int width, int height, const char* title) {
    SDL_Surface *final8 = SDL_CreateRGBSurfaceWithFormatFrom(
        (void*)grayscale_image,
        width,
        height,
        8,
        width * (int)sizeof(unsigned char),
        SDL_PIXELFORMAT_INDEX8
    );
    if (!final8) {
        fprintf(stderr, "Errore SDL_CreateRGBSurfaceWithFormatFrom (INDEX8): %s\n", SDL_GetError());
        return;
    }

    if (final8->format && final8->format->palette) {
        SDL_Color palette[256];
        for (int i = 0; i < 256; i++) {
            palette[i].r = i;
            palette[i].g = i;
            palette[i].b = i;
            palette[i].a = 255;
        }
        SDL_SetPaletteColors(final8->format->palette, palette, 0, 256);
    }

    SDL_Surface *final = SDL_ConvertSurfaceFormat(final8, SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(final8);
    if (!final) {
        fprintf(stderr, "Errore SDL_ConvertSurfaceFormat (to RGBA32): %s\n", SDL_GetError());
        return;
    }

    SDL_Window *window = create_window(title);
    if (!window) {
        fprintf(stderr, "Errore nella creazione della finestra: %s\n", SDL_GetError());
        SDL_FreeSurface(final);
        return;
    }

    SDL_Surface *screen = SDL_GetWindowSurface(window);
    if (!screen) {
        printf("Errore SDL_GetWindowSurface: %s\n", SDL_GetError());
        SDL_FreeSurface(final);
        SDL_DestroyWindow(window);
        return;
    }

    SDL_BlitScaled(final, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);
    SDL_Delay(SHOW_IMAGE_FOR);

    SDL_FreeSurface(final);
    SDL_DestroyWindow(window);
}



// RGB(A) version
void show_image(const uchar4* rgba_image, int width, int height, const char* title) {
    SDL_Surface *final = SDL_CreateRGBSurfaceWithFormatFrom(
        (void*)rgba_image,
        width,
        height,
        32,
        width * (int)sizeof(uchar4),
        SDL_PIXELFORMAT_RGBA32
    );
    if (!final) {
        fprintf(stderr, "Errore SDL_CreateRGBSurfaceWithFormatFrom (RGBA32): %s\n", SDL_GetError());
        return;
    }

    SDL_Window *window = create_window(title);
    if (!window) {
        fprintf(stderr, "Errore nella creazione della finestra: %s\n", SDL_GetError());
        SDL_FreeSurface(final);
        return;
    }

    SDL_Surface *screen = SDL_GetWindowSurface(window);
    if (!screen) {
        printf("Errore SDL_GetWindowSurface: %s\n", SDL_GetError());
        SDL_FreeSurface(final);
        SDL_DestroyWindow(window);
        return;
    }

    SDL_BlitScaled(final, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);
    SDL_Delay(SHOW_IMAGE_FOR);

    SDL_FreeSurface(final);
    SDL_DestroyWindow(window);
}