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

__global__ void gaussian_blur_kernel(uchar4 *input, uchar4 *output,int width, int height, int pitch4){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float3 sum = make_float3(0.f, 0.f, 0.f);

    for (int ky = -HALF; ky <= HALF; ky++) {
        int py = min(max(y + ky, 0), height - 1);
        for (int kx = -HALF; kx <= HALF; kx++) {
            int px = min(max(x + kx, 0), width - 1);

            uchar4 p = input[py * pitch4 + px];
            float w = d_kernel[(ky + HALF)*KERNEL_SIZE + (kx + HALF)];

            sum.x += p.x * w;
            sum.y += p.y * w;
            sum.z += p.z * w;
        }
    }

    uchar4 out;
    out.x = (unsigned char)sum.x;
    out.y = (unsigned char)sum.y;
    out.z = (unsigned char)sum.z;
    out.w = input[y * pitch4 + x].w;

    output[y * pitch4 + x] = out; // Preserva il canale alpha  
}

int main(int argc, char* argv[]) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
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
    
    SDL_Surface* dst = SDL_CreateRGBSurface(0, image->w, image->h, 32,
        image->format->Rmask, image->format->Gmask,
        image->format->Bmask, image->format->Amask);
        
    if (!dst) {
        fprintf(stderr, "Errore creazione superficie destinazione\n");
        return NULL;
    }

    SDL_LockSurface(image);
    SDL_LockSurface(dst);

    int width = image->w;
    int height = image->h;
    int pitch_bytes = image->pitch;
    int pitch = pitch_bytes / sizeof(uchar4);

    size_t bytes = pitch*height;
    uchar4 *d_input, *d_output;

    HANDLE_ERROR(cudaMalloc(&d_input, bytes*sizeof(uchar4)));
    HANDLE_ERROR(cudaMalloc(&d_output, bytes*sizeof(uchar4)));

    HANDLE_ERROR(cudaMemcpy(d_input, image->pixels, bytes*sizeof(uchar4), cudaMemcpyHostToDevice));

    // Genera e copia il kernel nella memoria costante della GPU
    init_gaussian_kernel();

    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 256 thread per blocco
    dim3 grid((width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y);

    gaussian_blur_kernel<<<grid, block>>>(d_input, d_output, width, height, pitch);
    checkCUDAError("Errore nella chiamata al kernel 'gaussian_blur_kernel()'");

    HANDLE_ERROR(cudaMemcpy(dst->pixels, d_output, bytes*sizeof(uchar4), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(d_input));
    HANDLE_ERROR(cudaFree(d_output));

    SDL_UnlockSurface(image);
    SDL_UnlockSurface(dst);

    if (!dst) {
        fprintf(stderr, "Errore durante il Gaussian blur\n");
        SDL_FreeSurface(image);
        SDL_DestroyWindow(windowNormal);
        SDL_DestroyWindow(windowBlur);
        IMG_Quit();
        SDL_Quit();
        return 1;
    }

    SDL_SaveBMP(dst, "immagine_sfocata.bmp");

    // Mostra entrambe le immagini contemporaneamente
    SDL_BlitScaled(image, NULL, screenNormal, NULL);
    SDL_UpdateWindowSurface(windowNormal);
    
    SDL_BlitScaled(dst, NULL, screenBlur, NULL);
    SDL_UpdateWindowSurface(windowBlur);
    
    // Attendi prima di chiudere
    SDL_Delay(4000);
    
    SDL_DestroyWindow(windowNormal);
    SDL_DestroyWindow(windowBlur);
    
    SDL_FreeSurface(dst);
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Execution time: %.3f ms\n", ms-4000);

    return 0;
}

