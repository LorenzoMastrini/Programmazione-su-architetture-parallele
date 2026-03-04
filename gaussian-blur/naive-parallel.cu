#include "../utils.cu"
#include <stdio.h>

#define KERNEL_SIZE 21
#define BLOCK_SIZE 16
#define HALF 10
#define SIGMA 5.0

__constant__ float d_kernel[KERNEL_SIZE*KERNEL_SIZE];

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

void init_gaussian_kernel() {
    static bool done = false;
    if (done) return;

    generate_gaussian_kernel();
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    done = true;
}

__global__ void gaussian_blur_kernel(unsigned char *input, unsigned char *output,int width, int height, int pitch){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    double r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;

    for (int ky = -HALF; ky <= HALF; ky++) {
        for (int kx = -HALF; kx <= HALF; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int pixel_index = py * pitch + px * 4;

            double weight = d_kernel[(ky + HALF) * KERNEL_SIZE + (kx + HALF)];
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
    
    // Genera e copia il kernel nella memoria costante della GPU
    init_gaussian_kernel();
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 256 thread per blocco
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    gaussian_blur_kernel<<<grid, block>>>(d_input, d_output, width, height, pitch);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Execution time: %.3f ms\n", ms);

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

    SDL_Surface* image = load_img("images/img.jpg");
    
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

