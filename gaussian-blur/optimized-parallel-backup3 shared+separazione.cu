#include "../utils.cu"
#include <stdio.h>

#define KERNEL_SIZE 21
#define BLOCK_SIZE 16
#define HALF 10
#define SIGMA 5.0

__constant__ float d_kernel_1d[KERNEL_SIZE];


float h_kernel_1d[KERNEL_SIZE];

void generate_gaussian_kernel() {
    double sum = 0.0;
    for (int i = -HALF; i <= HALF; i++) {
        double v = exp(-(i*i) / (2.0 * SIGMA * SIGMA));
        h_kernel_1d[i + HALF] = v;
        sum += v;
    }
    for (int i = 0; i < KERNEL_SIZE; i++)
        h_kernel_1d[i] /= sum;
}

void init_gaussian_kernel_once() {
    static bool initialized = false;
    if (initialized) return;

    generate_gaussian_kernel();
    HANDLE_ERROR(cudaMemcpyToSymbol(d_kernel_1d, h_kernel_1d, KERNEL_SIZE * sizeof(float)));

    initialized = true;
}

__global__ void gaussian_horizontal_shared(uchar4* __restrict__ input, uchar4* __restrict__ output,int width, int height){
    __shared__ uchar4 tile[BLOCK_SIZE][BLOCK_SIZE + 2 * HALF];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    int shared_x = tx + HALF;

    
    // Carica pixel centrale
    if (x < width && y < height)
        tile[ty][shared_x] = input[y * width + x];

    
    // Carica halo sinistro 
    if (tx < HALF) {
        int halo_x = max(x - HALF, 0);
        if (y < height)
            tile[ty][tx] = input[y * width + halo_x];
    }

    //Carica halo destro 
    if (tx >= BLOCK_SIZE - HALF) {
        int halo_x = min(x + HALF, width - 1);
        if (y < height)
            tile[ty][shared_x + HALF] = input[y * width + halo_x];
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    float r = 0.f, g = 0.f, b = 0.f;

    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        uchar4 pixel = tile[ty][tx + i];
        float w = d_kernel_1d[i];

        r += pixel.x * w;
        g += pixel.y * w;
        b += pixel.z * w;
    }

    uchar4 out;
    out.x = (unsigned char)r;
    out.y = (unsigned char)g;
    out.z = (unsigned char)b;
    out.w = tile[ty][shared_x].w;

    output[y * width + x] = out;
}

__global__ void gaussian_vertical_shared(uchar4* __restrict__ input,uchar4* __restrict__ output,int width, int height){
    __shared__ uchar4 tile[BLOCK_SIZE + 2 * HALF][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    int shared_y = ty + HALF;

    // centro
    if (x < width && y < height)
        tile[shared_y][tx] = input[y * width + x];

    // halo sopra
    if (ty < HALF) {
        int halo_y = max(y - HALF, 0);
        if (x < width)
            tile[ty][tx] = input[halo_y * width + x];
    }

    // halo sotto
    if (ty >= BLOCK_SIZE - HALF) {
        int halo_y = min(y + HALF, height - 1);
        if (x < width)
            tile[shared_y + HALF][tx] = input[halo_y * width + x];
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    float r = 0.f, g = 0.f, b = 0.f;

    #pragma unroll
    for (int i = 0; i < KERNEL_SIZE; i++) {
        uchar4 pixel = tile[ty + i][tx];
        float w = d_kernel_1d[i];

        r += pixel.x * w;
        g += pixel.y * w;
        b += pixel.z * w;
    }

    uchar4 out;
    out.x = (unsigned char)r;
    out.y = (unsigned char)g;
    out.z = (unsigned char)b;
    out.w = tile[shared_y][tx].w;

    output[y * width + x] = out;
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

    size_t num_pixels = width * height;
    size_t bytes = num_pixels * sizeof(uchar4);
    uchar4 *d_input, *d_temp;
    
    // Alloca memoria GPU
    HANDLE_ERROR(cudaMalloc(&d_input, bytes));
    HANDLE_ERROR(cudaMalloc(&d_temp, bytes));

    HANDLE_ERROR(cudaMemcpy(d_input, src->pixels, bytes, cudaMemcpyHostToDevice));
    init_gaussian_kernel_once();
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 256 thread per blocco
    dim3 grid((width + BLOCK_SIZE -1)/BLOCK_SIZE, (height + BLOCK_SIZE -1)/BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gaussian_horizontal_shared<<<grid, block>>>(d_input, d_temp, width, height);
    gaussian_vertical_shared<<<grid, block>>>(d_temp, d_input, width, height );
    checkCUDAError("Errore nella chiamata al kernel 'gaussian_blur_kernel()'");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Execution time: %.3f ms\n", ms);

    HANDLE_ERROR(cudaMemcpy(dst->pixels, d_input, bytes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(d_input));
    HANDLE_ERROR(cudaFree(d_temp));

    SDL_UnlockSurface(src);
    SDL_UnlockSurface(dst);

    return dst;
}

int main(int argc, char* argv[]) {
    init_SDL();
    cudaFree(0);
    
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

