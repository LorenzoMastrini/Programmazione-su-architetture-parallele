#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>

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



// Conversione in-place di un'immagine da RGB a scala di grigi
void rgb_to_grayscale(SDL_Surface *img) {
    // Blocca la superficie per accedere ai pixel
    SDL_LockSurface(img);

    Uint32 *pixels = (Uint32 *)img->pixels;
    int pitch = img->pitch / 4; // numero di Uint32 per riga (pitch = #pixel per riga)

    for (int y = 0; y < img->h; y++) {
        for (int x = 0; x < img->w; x++) {
            Uint32 p = pixels[y * pitch + x]; // accedo a ogni pixel

            Uint8 a = (p >> 24) & 0xFF;
            Uint8 r = (p >> 16) & 0xFF;
            Uint8 g = (p >> 8) & 0xFF;
            Uint8 b = p & 0xFF;

            // Scala di grigi
            Uint8 gray = 0.3 * r + 0.59 * g + 0.11 * b;
            pixels[y * pitch + x] = (a << 24) | (gray << 16) | (gray << 8) | gray;
        }
    }

    SDL_UnlockSurface(img);
}



int main() {
    init_SDL();
    
    SDL_Window *window = SDL_CreateWindow("Scala di grigi con SDL_Surface",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          800, 600, 0);

    // Ottieni la superficie della finestra (dove disegneremo)
    SDL_Surface *screen = SDL_GetWindowSurface(window);

    // Carica l'immagine in una SDL_Surface
    SDL_Surface *image = IMG_Load("img.jpg");
    if (!image) {
        printf("Errore IMG_Load: %s\n", IMG_GetError());
        return 1;
    }

    // Converti nel formato della finestra per evitare problemi di formato pixel
    SDL_Surface *converted = SDL_ConvertSurfaceFormat(image, SDL_PIXELFORMAT_ARGB8888, 0); // ogni canale è composto da 8 bit (1 byte) => ogni pixel da 32 bit (4 byte)
    SDL_FreeSurface(image);

    rgb_to_grayscale(converted);

    // Copia la superficie nella finestra (scalandone correttamente le dimensioni)
    SDL_BlitScaled(converted, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);

    SDL_Delay(3000); // Mostra per 3 secondi

    // Free everything
    SDL_FreeSurface(converted);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();
    return 0;
}