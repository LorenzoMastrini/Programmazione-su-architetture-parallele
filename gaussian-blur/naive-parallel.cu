#include "../utils.h"
#include <stdio.h>

#define KERNEL_SIZE 5

__constant__ double d_kernel[KERNEL_SIZE*KERNEL_SIZE];


// Kernel gaussiano 5x5 pre-calcolato e normalizzato
double h_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
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

__global__ void gaussian_blur_kernel(unsigned char *input, unsigned char *output,int width, int height, int pitch){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int half = KERNEL_SIZE / 2;

    double r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int pixel_index = py * pitch + px * 4;

            double weight = d_kernel[(ky + half) * KERNEL_SIZE + (kx + half)];
            r_sum += input[pixel_index + 0] * weight;
            g_sum += input[pixel_index + 1] * weight;
            b_sum += input[pixel_index + 2] * weight;
        }
    }

    int out_index = y * pitch + x * 4;
    output[out_index + 0] = (unsigned char)r_sum;
    output[out_index + 1] = (unsigned char)g_sum;
    output[out_index + 2] = (unsigned char)b_sum;
    output[out_index + 3] = input[y * pitch + x * 4 + 3]; // Preserva il canale alpha  
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

    int width = src->w;
    int height = src->h;
    int pitch = src->pitch;

    size_t bytes = height * pitch;
    unsigned char *d_input, *d_output;

    HANDLE_ERROR(cudaMalloc(&d_input, bytes));
    HANDLE_ERROR(cudaMalloc(&d_output, bytes));

    HANDLE_ERROR(cudaMemcpy(d_input, src->pixels, bytes, cudaMemcpyHostToDevice));
    
    // Converti h_kernel da 2D a 1D per copiarlo nella memoria costante
    double h_kernel_1d[KERNEL_SIZE * KERNEL_SIZE];
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            h_kernel_1d[i * KERNEL_SIZE + j] = h_kernel[i][j];
        }
    }
    HANDLE_ERROR(cudaMemcpyToSymbol(d_kernel, h_kernel_1d, KERNEL_SIZE * KERNEL_SIZE * sizeof(double)));    
    
    dim3 block(16, 16); // 256 thread per blocco
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    gaussian_blur_kernel<<<grid, block>>>(d_input, d_output, width, height, pitch);
    cudaDeviceSynchronize();
    checkCUDAError("Errore nella chiamata al kernel 'gaussian_blur_kernel()'");

    HANDLE_ERROR(cudaMemcpy(dst->pixels, d_output, bytes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(d_input));
    HANDLE_ERROR(cudaFree(d_output));

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

