#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define KERNEL_SIZE 5

// Initialize SDL modules
int init_SDL() {
    // Carica il modulo grafico della libreria
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("Errore SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    // Formati di immagini supportate
    if (!(IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG))) {
        printf("Errore IMG_Init: %s\n", IMG_GetError());
        return 1;
    }

    return 0;
}

// Kernel gaussiano 5x5 pre-calcolato e normalizzato
double kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1.0/273,  4.0/273,  7.0/273,  4.0/273, 1.0/273},
    {4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273},
    {7.0/273, 26.0/273, 41.0/273, 26.0/273, 7.0/273},
    {4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273},
    {1.0/273,  4.0/273,  7.0/273,  4.0/273, 1.0/273}
};

// Funzioni per leggere e scrivere pixel da SDL_Surface 32bit
Uint32 get_pixel(SDL_Surface *surface, int x, int y) {
    if (x < 0) x = 0;
    if (x >= surface->w) x = surface->w - 1;
    if (y < 0) y = 0;
    if (y >= surface->h) y = surface->h - 1;
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * 4;
    return *(Uint32 *)p;
}

void put_pixel(SDL_Surface *surface, int x, int y, Uint32 pixel) {
    if (x < 0 || x >= surface->w || y < 0 || y >= surface->h) return;
    Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * 4;
    *(Uint32 *)p = pixel;
}

// Funzione di Gaussian blur per immagine RGB
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
            // Se il kernel uscirebbe fuori, copia il pixel originale
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

                    Uint8 r, g, b, a;
                    SDL_GetRGBA(pixel, src->format, &r, &g, &b, &a);

                    double weight = kernel[ky + half][kx + half];
                    r_sum += r * weight;
                    g_sum += g * weight;
                    b_sum += b * weight;
                }
            }
            Uint8 r_blur = (Uint8)r_sum;
            Uint8 g_blur = (Uint8)g_sum;
            Uint8 b_blur = (Uint8)b_sum;
            Uint8 a = 255;

            Uint32 new_pixel = SDL_MapRGBA(dst->format, r_blur, g_blur, b_blur, a);
            put_pixel(dst, x, y, new_pixel);
        }
    }

    SDL_UnlockSurface(src);
    SDL_UnlockSurface(dst);

    return dst;
}

int main(int argc, char* argv[]) {
    init_SDL();
    
    SDL_Window *windowNormal = SDL_CreateWindow("Gaussian Blur",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        0, 0, SDL_WINDOW_FULLSCREEN_DESKTOP);
    
    SDL_Window *windowBlur = SDL_CreateWindow("Gaussian Blur",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        0, 0, SDL_WINDOW_FULLSCREEN_DESKTOP);
    
    if (!windowNormal) {
        fprintf(stderr, "Impossibile creare la finestra: %s\n", SDL_GetError());
        IMG_Quit();
        SDL_Quit();
        return 1;
    }
    
    SDL_Surface *screenNormal = SDL_GetWindowSurface(windowNormal);
    if (!screenNormal) {
        fprintf(stderr, "Impossibile ottenere la surface della finestra: %s\n", SDL_GetError());
        SDL_DestroyWindow(windowNormal);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_Surface *screenBlur = SDL_GetWindowSurface(windowBlur);
    if (!screenBlur) {
        fprintf(stderr, "Impossibile ottenere la surface della finestra: %s\n", SDL_GetError());
        SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_Surface* image = IMG_Load("../images/Lenna_(test_image).png");
    if (!image) {
        fprintf(stderr, "Impossibile caricare immagine: %s\n", IMG_GetError());
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    // Converti l'immagine in 32-bit RGBA per garantire 4 byte per pixel
    SDL_Surface* image32 = SDL_ConvertSurfaceFormat(image, SDL_PIXELFORMAT_RGBA32, 0);
    if (!image32) {
        fprintf(stderr, "Impossibile convertire l'immagine in RGBA32: %s\n", SDL_GetError());
        SDL_FreeSurface(image);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }
    SDL_FreeSurface(image);
    image = image32;

    SDL_Surface* blurred = gaussian_blur(image);
    if (!blurred) {
        fprintf(stderr, "Errore durante il Gaussian blur\n");
        SDL_FreeSurface(image);
        SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_SaveBMP(blurred, "immagine_sfocata.bmp");

    SDL_BlitScaled(image, NULL, screenNormal, NULL);
    SDL_UpdateWindowSurface(windowNormal);
    SDL_Delay(4000);
    SDL_DestroyWindow(windowNormal);
    
    SDL_BlitScaled(blurred, NULL, screenBlur, NULL);
    SDL_UpdateWindowSurface(windowBlur);
    SDL_Delay(4000);
    SDL_DestroyWindow(windowBlur);
    
    SDL_FreeSurface(blurred);
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
}
