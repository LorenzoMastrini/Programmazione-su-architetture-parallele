#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <stdio.h>

int main() {
    // Carica solo il modulo grafico
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("Errore SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    // Estensioni di immagini supportate
    if (!(IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG))) {
        printf("Errore IMG_Init: %s\n", IMG_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow("Da RGB a scala di grigi",
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, // posizione finestra
                                          800, 600, 0); // dimensioni (e nessun flag)
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    // Carica immagine in una SDL_Surface
    SDL_Surface *image = IMG_Load("img.jpg");
    if (!image) {
        printf("Errore IMG_Load: %s\n", IMG_GetError());
        return 1;
    }

    int width = image->w;
    int height = image->h;

    // Crea una texture streaming con lo stesso formato
    SDL_Texture *texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_ARGB8888,
                                             SDL_TEXTUREACCESS_STREAMING, // necessario che la texture sia streaming per poterla modificare
                                             width, height);
    if (!texture) {
        printf("Errore SDL_CreateTexture: %s\n", SDL_GetError());
        SDL_FreeSurface(image);
        return 1;
    }

    // Copia i pixel dell’immagine nella texture streaming
    void *pixels;
    int pitch;
    SDL_LockTexture(texture, NULL, &pixels, &pitch);

    // Convertiamo l’immagine nel formato ARGB8888 per essere sicuri che i colori coincidano
    SDL_Surface *converted = SDL_ConvertSurfaceFormat(image, SDL_PIXELFORMAT_ARGB8888, 0);
    SDL_FreeSurface(image); // non serve più l’originale

    Uint32 *src = (Uint32 *)converted->pixels;
    Uint32 *dst = (Uint32 *)pixels;
    int src_pitch = converted->pitch / 4;
    int dst_pitch = pitch / 4;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            dst[y * dst_pitch + x] = src[y * src_pitch + x];
        }
    }

    SDL_UnlockTexture(texture);

    // Modifichiamo i pixel direttamente nella texture
    SDL_LockTexture(texture, NULL, &pixels, &pitch);
    Uint32 *pix = (Uint32 *)pixels;
    int line = pitch / 4;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Uint32 p = pix[y * line + x];
            Uint8 a = (p >> 24) & 0xFF;
            Uint8 r = (p >> 16) & 0xFF;
            Uint8 g = (p >> 8) & 0xFF;
            Uint8 b = p & 0xFF;

            // Trasforma i colori RGB in grayscale tramite formula
            Uint8 gray = 0.3*r + 0.59*g + 0.11*b; // pesare i colori in base alla percezione dell'occhio umano
            pix[y * line + x] = (a << 24) | (gray << 16) | (gray << 8) | gray;
        }
    }
    
    SDL_UnlockTexture(texture);
    SDL_FreeSurface(converted);

    // === 5️⃣ Mostra la texture nella finestra
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    SDL_Delay(5000); // mostra per 5 secondi

    // === 6️⃣ Pulisci tutto
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();
    return 0;
}
