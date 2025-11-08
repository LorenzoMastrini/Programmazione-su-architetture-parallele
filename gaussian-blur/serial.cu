#include <stdio.h>
#include "../utils.h"

#define KERNEL_SIZE 5


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

int main(int argc, char* argv[]) {
    init_SDL();
    
    SDL_Window *windowNormal = create_window("Gaussian Blur");
    
    SDL_Window *windowBlur = create_window("Gaussian Blur");
    
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

    SDL_Surface* image = load_img("../images/Lenna_(test_image).png");

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
