#include "../utils.h"
#include <stdio.h>

#define KERNEL_SIZE 21
#define BLOCK_SIZE 32
#define HALF 10
#define SIGMA 5.0

float *d_kernel_1d = nullptr;

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

void init_gaussian_kernel() {
    static bool initialized = false;
    if (initialized) return;

    generate_gaussian_kernel();
    if (d_kernel_1d == nullptr) {
        HANDLE_ERROR(cudaMalloc(&d_kernel_1d, KERNEL_SIZE * sizeof(float)));
    }
    HANDLE_ERROR(cudaMemcpy(d_kernel_1d, h_kernel_1d, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    initialized = true;
}

__global__ void gaussian_horizontal(cudaTextureObject_t texObj, uchar4 __restrict__ *output, const float *kernel, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float r = 0, g = 0, b = 0;

    for (int i = 0; i < KERNEL_SIZE; i++) {
        int k = i - HALF;

        int px = x + k;
        uchar4 pixel = tex2D<uchar4>(texObj, px, y);

        float w = kernel[i];

        r += pixel.x * w;
        g += pixel.y * w;
        b += pixel.z * w;
    }


    uchar4 out;
    out.x = (unsigned char)r;
    out.y = (unsigned char)g;
    out.z = (unsigned char)b;
    out.w = tex2D<uchar4>(texObj, x, y).w;

    output[y*width+x] = out;
}

__global__ void gaussian_vertical(cudaTextureObject_t texObj, uchar4 __restrict__ *output, const float *kernel, int width, int height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float r = 0, g = 0, b = 0;

    
    for (int i = 0; i < KERNEL_SIZE; i++) {
        int k = i - HALF;
        int py = y + k;
        uchar4 pixel = tex2D<uchar4>(texObj, x, py);
        float w = kernel[i];

        r += pixel.x * w;
        g += pixel.y * w;
        b += pixel.z * w;
    }

    uchar4 out;
    out.x = (unsigned char)r;
    out.y = (unsigned char)g;
    out.z = (unsigned char)b;
    out.w = tex2D<uchar4>(texObj, x, y).w;
    
    output[y*width+x] = out;
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    SDL_LockSurface(src);
    SDL_LockSurface(dst);

    int width = src->w;
    int height = src->h;

    size_t num_pixels = width * height;
    size_t bytes = num_pixels * sizeof(uchar4);
    uchar4 *d_input, *d_temp;

    cudaEventRecord(start);
    HANDLE_ERROR(cudaMalloc(&d_input, bytes));
    HANDLE_ERROR(cudaMalloc(&d_temp, bytes));

    HANDLE_ERROR(cudaMemcpy(d_input, src->pixels, bytes, cudaMemcpyHostToDevice));
    init_gaussian_kernel();

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_input;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = width * sizeof(uchar4);

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texInput, texTemp;

    // texture su d_input
    resDesc.res.pitch2D.devPtr = d_input;
    cudaCreateTextureObject(&texInput, &resDesc, &texDesc, NULL);

    // texture su d_temp
    resDesc.res.pitch2D.devPtr = d_temp;
    cudaCreateTextureObject(&texTemp, &resDesc, &texDesc, NULL);

    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 256 thread per blocco
    dim3 grid((width + BLOCK_SIZE -1)/BLOCK_SIZE, (height + BLOCK_SIZE -1)/BLOCK_SIZE);
    
    gaussian_horizontal<<<grid, block>>>(texInput, d_temp, d_kernel_1d, width, height);
    checkCUDAError("Errore nella chiamata al kernel 'gaussian_horizontal()'");
    gaussian_vertical<<<grid, block>>>(texTemp, d_input, d_kernel_1d, width, height);
    checkCUDAError("Errore nella chiamata al kernel 'gaussian_vertical()'");
 
    HANDLE_ERROR(cudaMemcpy(dst->pixels, d_input, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Execution time: %.3f ms\n", ms);

    HANDLE_ERROR(cudaDestroyTextureObject(texInput));
    HANDLE_ERROR(cudaDestroyTextureObject(texTemp));
    HANDLE_ERROR(cudaFree(d_input));
    HANDLE_ERROR(cudaFree(d_temp));

    SDL_UnlockSurface(src);
    SDL_UnlockSurface(dst);

    return dst;
}

int main(int argc, char* argv[]) {
    init_SDL();
    cudaFree(0);
    
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
    
    SDL_SetWindowPosition(windowNormal, 0, 0);
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

    SDL_BlitScaled(image, NULL, screenNormal, NULL);
    SDL_UpdateWindowSurface(windowNormal);
    
    SDL_BlitScaled(blurred, NULL, screenBlur, NULL);
    SDL_UpdateWindowSurface(windowBlur);

    SDL_Delay(4000);
    
    SDL_DestroyWindow(windowNormal);
    SDL_DestroyWindow(windowBlur);
    
    SDL_FreeSurface(blurred);
    SDL_FreeSurface(image);
    IMG_Quit();
    SDL_Quit();


    return 0;
}

