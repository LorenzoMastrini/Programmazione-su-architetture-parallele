#include <stdio.h>
#include "../utils.h"

#define KERNEL_SIZE 21
#define BLOCK_SIZE 16
#define HALF 10
#define SIGMA 5.0

float h_kernel[KERNEL_SIZE * KERNEL_SIZE];

// Genera il kernel gaussiano normalizzato a runtime
void generate_gaussian_kernel() {
    double sum = 0.0;
    
    for (int y = -HALF; y <= HALF; y++) {
        for (int x = -HALF; x <= HALF; x++) {
            double value = exp(-(x*x + y*y) / (2.0 * SIGMA * SIGMA));
            h_kernel[(y + HALF) * KERNEL_SIZE + (x + HALF)] = value;
            sum += value;
        }
    }
    
    // Normalizza
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        h_kernel[i] /= sum;
    }
}

// Funzione di Gaussian blur per immagine RGB
SDL_Surface* gaussian_blur(SDL_Surface* src) {
    SDL_Surface* dst = SDL_CreateRGBSurface(
        0,
        src->w, src->h, 32,
        src->format->Rmask,
        src->format->Gmask,
        src->format->Bmask,
        src->format->Amask
    );

    if (!dst) {
        fprintf(stderr, "Errore creazione superficie destinazione\n");
        return NULL;
    }

    SDL_LockSurface(src);
    SDL_LockSurface(dst);

    int width = src->w;
    int height = src->h;
    int pitch = src->pitch;

    unsigned char* input  = (unsigned char*)src->pixels;
    unsigned char* output = (unsigned char*)dst->pixels;

    generate_gaussian_kernel();

    // -------------------------------------------------- CUDA --------------------------------------------------
    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            double r_sum = 0.0;
            double g_sum = 0.0;
            double b_sum = 0.0;

            for (int ky = -HALF; ky <= HALF; ky++) {
                for (int kx = -HALF; kx <= HALF; kx++) {

                    int px = x + kx;
                    int py = y + ky;

                    // Clamp bordi (stessa logica CUDA)
                    if (px < 0) px = 0;
                    if (py < 0) py = 0;
                    if (px >= width)  px = width - 1;
                    if (py >= height) py = height - 1;

                    int pixel_index = py * pitch + px * 4;
                    double weight = h_kernel[(ky + HALF) * KERNEL_SIZE + (kx + HALF)];

                    r_sum += input[pixel_index + 0] * weight;
                    g_sum += input[pixel_index + 1] * weight;
                    b_sum += input[pixel_index + 2] * weight;
                }
            }

            int out_index = y * pitch + x * 4;
            output[out_index + 0] = (unsigned char)r_sum;
            output[out_index + 1] = (unsigned char)g_sum;
            output[out_index + 2] = (unsigned char)b_sum;
            output[out_index + 3] = input[out_index + 3]; // alpha invariato
        }
    }

    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Total CUDA time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // -------------------------------------------------- END CUDA --------------------------------------------------

    
    SDL_UnlockSurface(src);
    SDL_UnlockSurface(dst);

    return dst;
}

int main(int argc, char* argv[]) {
    init_SDL();

    // Ottieni le dimensioni dello schermo per posizionare le finestre
    SDL_DisplayMode dm;
    if (SDL_GetDesktopDisplayMode(0, &dm) != 0) {
        fprintf(stderr, "Errore SDL_GetDesktopDisplayMode: %s\n", SDL_GetError());
        IMG_Quit();
        SDL_Quit();
        return 1;
    }
    
    SDL_Window *windowNormal = create_window("Immagine Originale");
    SDL_Window *windowBlur = create_window("Immagine Sfocata");
    
    if (!windowNormal || !windowBlur) {
        fprintf(stderr, "Impossibile creare le finestre: %s\n", SDL_GetError());
        if (windowNormal) SDL_DestroyWindow(windowNormal);
        if (windowBlur) SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }
    
    // Posiziona le finestre una di fianco all'altra
    // Prima finestra a sinistra
    SDL_SetWindowPosition(windowNormal, 0, 0);
    // Seconda finestra a destra (metà schermo)
    SDL_SetWindowPosition(windowBlur, dm.w / 2, 0);
    
    SDL_Surface *screenNormal = SDL_GetWindowSurface(windowNormal);
    if (!screenNormal) {
        fprintf(stderr, "Impossibile ottenere la surface della finestra: %s\n", SDL_GetError());
        SDL_DestroyWindow(windowNormal);
        SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_Surface *screenBlur = SDL_GetWindowSurface(windowBlur);
    if (!screenBlur) {
        fprintf(stderr, "Impossibile ottenere la surface della finestra: %s\n", SDL_GetError());
        SDL_DestroyWindow(windowNormal);
        SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_Surface* image = load_img(IMG_PATH);
    generate_gaussian_kernel();

    SDL_Surface* blurred = gaussian_blur(image);
    if (!blurred) {
        fprintf(stderr, "Errore durante il Gaussian blur\n");
        SDL_FreeSurface(image);
        SDL_DestroyWindow(windowNormal);
        SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_SaveBMP(blurred, "immagine_sfocata.bmp");

    // Mostra entrambe le immagini contemporaneamente
    SDL_BlitScaled(image, NULL, screenNormal, NULL);
    SDL_UpdateWindowSurface(windowNormal);
    
    SDL_BlitScaled(blurred, NULL, screenBlur, NULL);
    SDL_UpdateWindowSurface(windowBlur);
    
    // Attendi prima di chiudere
    SDL_Delay(4000);
    
    SDL_DestroyWindow(windowNormal);
    SDL_DestroyWindow(windowBlur);
    
    SDL_FreeSurface(blurred);
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
}