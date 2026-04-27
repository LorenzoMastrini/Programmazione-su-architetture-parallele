#include <stdio.h>
#include "../utils.h"

#define KERNEL_SIZE 21
#define HALF 10
#define SIGMA 5.0

float h_kernel[KERNEL_SIZE];

// Genera il kernel gaussiano normalizzato a runtime
void generate_gaussian_kernel() {
    double sum = 0.0;
    for (int i = -HALF; i <= HALF; i++) {
        double v = exp(-(i*i) / (2.0 * SIGMA * SIGMA));
        h_kernel[i + HALF] = v;
        sum += v;
    }
    for (int i = 0; i < KERNEL_SIZE; i++)
        h_kernel[i] /= sum;
}

// Funzione di Gaussian blur per immagine RGB
SDL_Surface* gaussian_blur(SDL_Surface* src) {
    SDL_Surface* tmp = SDL_CreateRGBSurface(
        0, src->w, src->h, 32,
        src->format->Rmask,
        src->format->Gmask,
        src->format->Bmask,
        src->format->Amask
    );

    SDL_Surface* dst = SDL_CreateRGBSurface(
        0, src->w, src->h, 32,
        src->format->Rmask,
        src->format->Gmask,
        src->format->Bmask,
        src->format->Amask
    );

    if (!tmp || !dst) {
        fprintf(stderr, "Errore creazione superfici\n");
        return NULL;
    }

    SDL_LockSurface(src);
    SDL_LockSurface(tmp);
    SDL_LockSurface(dst);

    int width = src->w;
    int height = src->h;

    uchar4* input  = (uchar4*)src->pixels;
    uchar4* buffer = (uchar4*)tmp->pixels;
    uchar4* output = (uchar4*)dst->pixels;

    generate_gaussian_kernel();

    // PASS 1: ORIZZONTALE
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            float4 sum = {0,0,0,0};

            for (int k = -HALF; k <= HALF; k++) {
                int px = x + k;

                if (px < 0) px = 0;
                if (px >= width) px = width - 1;

                uchar4 p = input[y * width + px];
                float w = h_kernel[k + HALF];

                sum.x += p.x * w;
                sum.y += p.y * w;
                sum.z += p.z * w;
                sum.w += p.w * w;
            }

            uchar4 out;
            out.x = (unsigned char)sum.x;
            out.y = (unsigned char)sum.y;
            out.z = (unsigned char)sum.z;
            out.w = (unsigned char)sum.w;

            buffer[y * width + x] = out;
        }
    }

    // PASS 2: VERTICALE
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            float4 sum = {0,0,0,0};

            for (int k = -HALF; k <= HALF; k++) {
                int py = y + k;

                if (py < 0) py = 0;
                if (py >= height) py = height - 1;

                uchar4 p = buffer[py * width + x];
                float w = h_kernel[k + HALF];

                sum.x += p.x * w;
                sum.y += p.y * w;
                sum.z += p.z * w;
                sum.w += p.w * w;
            }

            uchar4 out;
            out.x = (unsigned char)sum.x;
            out.y = (unsigned char)sum.y;
            out.z = (unsigned char)sum.z;
            out.w = (unsigned char)sum.w;

            output[y * width + x] = out;
        }
    }

    SDL_UnlockSurface(src);
    SDL_UnlockSurface(tmp);
    SDL_UnlockSurface(dst);

    SDL_FreeSurface(tmp);

    return dst;
}

int main(int argc, char* argv[]) {
    init_SDL();

    SDL_Surface* image = load_img(IMG_PATH);
    
    generate_gaussian_kernel();

    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    SDL_Surface* blurred = gaussian_blur(image);

    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    show_image(reinterpret_cast<const uchar4*>(image->pixels), image->w, image->h, "Immagine Originale");
    show_image(reinterpret_cast<const uchar4*>(blurred->pixels), blurred->w, blurred->h, "Immagine Sfocata");
    
    SDL_FreeSurface(blurred);
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();

    return 0;
}