#include <stdio.h>
#include <math.h>
#include "../utils.h"


Uint8 get_gray(Uint32 pixel, SDL_PixelFormat *format) {
    Uint8 r, g, b;
    SDL_GetRGB(pixel, format, &r, &g, &b);
    return (Uint8)(0.3*r + 0.59*g + 0.11*b);
}



Uint32 make_pixel(SDL_PixelFormat *format, Uint8 gray) {
    return SDL_MapRGB(format, gray, gray, gray);
}



int main(int argc, char *argv[]) {
    init_SDL();

    SDL_Surface *image = load_img("./images/img.jpg");

    SDL_Surface *edges = SDL_CreateRGBSurface(0, image->w, image->h, 32,
                                              0x00FF0000,
                                              0x0000FF00,
                                              0x000000FF,
                                              0xFF000000);
                                            
    if (!edges) {
        printf("Errore SDL_CreateRGBSurface: %s\n", SDL_GetError());
        return 1;
    }

    int w = image->w;
    int h = image->h;
    SDL_LockSurface(image);
    SDL_LockSurface(edges);

    Uint32 *pixels = (Uint32 *) image->pixels;
    Uint32 *out = (Uint32 *) edges->pixels;

    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int sobel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int gx = 0;
            int gy = 0;

            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    Uint8 gray = get_gray(pixels[(y + j) * w + (x + i)], image->format);
                    gx += sobel_x[j + 1][i + 1] * gray;
                    gy += sobel_y[j + 1][i + 1] * gray;
                }
            }

            int g = (int) sqrt(gx * gx + gy * gy);
            if (g > 255) g = 255;
            out[y * w + x] = make_pixel(edges->format, (Uint8)g);
        }
    }

    SDL_UnlockSurface(image);
    SDL_UnlockSurface(edges);

    // Create the window to show the image
    SDL_Window *window = create_window("Sobel edge detection");

    // Ottieni la superficie della finestra (dove disegneremo)
    SDL_Surface *screen = SDL_GetWindowSurface(window);
    if (!screen) {
        printf("Errore SDL_GetWindowSurface: %s\n", SDL_GetError());
        return 1;
    }

    // Copia la superficie nella finestra (scalandone correttamente le dimensioni)
    SDL_BlitScaled(edges, NULL, screen, NULL);
    SDL_UpdateWindowSurface(window);

    SDL_Delay(3000); // Mostra per 3 secondi

    // Free everything
    SDL_FreeSurface(edges);
    SDL_FreeSurface(image);
    SDL_DestroyWindow(window);
    IMG_Quit();
    SDL_Quit();

    return 0;
}