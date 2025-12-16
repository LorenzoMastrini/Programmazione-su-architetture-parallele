#include "../utils.h"
#include <stdio.h>

#define KERNEL_SIZE 5
#define BLOCK_SIZE 32
#define HALF 2

__constant__ float d_kernel_1d[KERNEL_SIZE];


// Kernel gaussiano 5x5 pre-calcolato e normalizzato (array 1D in ordine row-major)
/*double h_kernel[KERNEL_SIZE * KERNEL_SIZE] = {
    1.0/273,  4.0/273,  7.0/273,  4.0/273, 1.0/273,
    4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
    7.0/273, 26.0/273, 41.0/273, 26.0/273, 7.0/273,
    4.0/273, 16.0/273, 26.0/273, 16.0/273, 4.0/273,
    1.0/273,  4.0/273,  7.0/273,  4.0/273, 1.0/273
};*/

float h_kernel_1d[KERNEL_SIZE] = {
    1.0f/16, 4.0f/16, 6.0f/16, 4.0f/16, 1.0f/16
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

/*__global__ void gaussian_blur_kernel(unsigned char *input, unsigned char *output,int width, int height, int pitch){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int half = KERNEL_SIZE / 2;

    float r_sum = 0.0, g_sum = 0.0, b_sum = 0.0;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int pixel_index = py * pitch + px * 4;

            float weight = d_kernel[(ky + half) * KERNEL_SIZE + (kx + half)];
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
}*/
/*__global__ void gaussian_blur_kernel_shared(unsigned char *input, unsigned char *output, int width, int height, int pitch){
    // Padding per evitare bank conflicts: aggiungiamo 1 elemento per allineare a 4 bytes
    __shared__ unsigned char tile[BLOCK_SIZE + 2*HALF][BLOCK_SIZE + 2*HALF][4]; // RGB + padding, alpha copiato dopo

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    // Coord nel tile condiviso
    int tile_x = tx + HALF;
    int tile_y = ty + HALF;

    // Carica i pixel centrali nel tile condiviso
    for (int c = 0; c < 3; c++) {
        if (x < width && y < height)
            tile[tile_y][tile_x][c] = input[y*pitch + x*4 + c];
        else
            tile[tile_y][tile_x][c] = 0;
    }

    // Carica i bordi (halo)
    if (tx < HALF) {
        for (int c = 0; c < 3; c++) {
            // sinistra
            int xx = max(x - HALF, 0);
            int yy = y;
            tile[tile_y][tile_x - HALF][c] = (yy < height) ? input[yy*pitch + xx*4 + c] : 0;
            // destra
            xx = min(x + BLOCK_SIZE, width-1);
            tile[tile_y][tile_x + BLOCK_SIZE][c] = (yy < height) ? input[yy*pitch + xx*4 + c] : 0;
        }
    }

    if (ty < HALF) {
        for (int c = 0; c < 3; c++) {
            // sopra
            int xx = x;
            int yy = max(y - HALF, 0);
            tile[tile_y - HALF][tile_x][c] = (xx < width) ? input[yy*pitch + xx*4 + c] : 0;
            // sotto
            yy = min(y + BLOCK_SIZE, height-1);
            tile[tile_y + BLOCK_SIZE][tile_x][c] = (xx < width) ? input[yy*pitch + xx*4 + c] : 0;
        }
    }

    // Carica gli angoli del bordo (halo diagonale)
    if (tx < HALF && ty < HALF) {
        for (int c = 0; c < 3; c++) {
            int xx = max(x - HALF, 0);
            int yy = max(y - HALF, 0);
            tile[tile_y - HALF][tile_x - HALF][c] = (xx < width && yy < height) ? input[yy*pitch + xx*4 + c] : 0;

            xx = min(x + BLOCK_SIZE, width-1);
            yy = max(y - HALF, 0);
            tile[tile_y - HALF][tile_x + BLOCK_SIZE][c] = (xx < width && yy < height) ? input[yy*pitch + xx*4 + c] : 0;

            xx = max(x - HALF, 0);
            yy = min(y + BLOCK_SIZE, height-1);
            tile[tile_y + BLOCK_SIZE][tile_x - HALF][c] = (xx < width && yy < height) ? input[yy*pitch + xx*4 + c] : 0;

            xx = min(x + BLOCK_SIZE, width-1);
            yy = min(y + BLOCK_SIZE, height-1);
            tile[tile_y + BLOCK_SIZE][tile_x + BLOCK_SIZE][c] = (xx < width && yy < height) ? input[yy*pitch + xx*4 + c] : 0;
        }
    }

    __syncthreads(); // sincronizza tutti i thread del blocco

    if (x >= width || y >= height) return;

    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
    // Convoluzione 5x5 usando dati in shared memory
    for (int ky = -HALF; ky <= HALF; ky++) {
        for (int kx = -HALF; kx <= HALF; kx++) {
            float weight = d_kernel[(ky + HALF) * KERNEL_SIZE + (kx + HALF)];
            r_sum += tile[tile_y + ky][tile_x + kx][0] * weight;
            g_sum += tile[tile_y + ky][tile_x + kx][1] * weight;
            b_sum += tile[tile_y + ky][tile_x + kx][2] * weight;
        }
    }

    int out_index = y*pitch + x*4;
    output[out_index + 0] = (unsigned char)r_sum;
    output[out_index + 1] = (unsigned char)g_sum;
    output[out_index + 2] = (unsigned char)b_sum;
    output[out_index + 3] = input[out_index + 3]; // preserva alpha
}*/

__global__ void gaussian_horizontal_shared(uchar4 *input, uchar4 *output, int width, int height, int pitch){
    __shared__ uchar4 tile[BLOCK_SIZE + 2*HALF];

    int tx = threadIdx.x;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y;  // ogni blocco lavora su UNA sola riga

    if (y >= height) return;

    int tile_x = tx + HALF;

    // Carica pixel centrale
    if (x < width) {
        int idx = y * pitch + x;
        tile[tile_x] = input[idx];
    } else {
        tile[tile_x] = make_uchar4(0,0,0,0);
    }

    // Halo sinistro
    if (tx < HALF) {
        int xx = max(x - HALF, 0);
        int idx = y * pitch + xx;
        tile[tx] = input[idx];
    }

    // Halo destro
    if (tx >= BLOCK_SIZE - HALF) {
        int xx = min(x + HALF, width - 1);
        int idx = y * pitch + xx;
        tile[tile_x + HALF] = input[idx];
    }

    __syncthreads();

    if (x >= width) return;

    float r=0, g=0, b=0;

    for (int k = -HALF; k <= HALF; k++) {
        float w = d_kernel_1d[k + HALF];
        uchar4 p = tile[tile_x + k];
        r += p.x * w;
        g += p.y * w;
        b += p.z * w;
    }

    int out = y * pitch + x;
    uchar4 outp = input[out];        
    outp.x = (unsigned char)r;
    outp.y = (unsigned char)g;
    outp.z = (unsigned char)b;
    output[out] = outp;
}

__global__ void gaussian_vertical_shared(uchar4 *input, uchar4 *output, int width, int height, int pitch){
    __shared__ uchar4 tile[BLOCK_SIZE + 2*HALF];

    int ty = threadIdx.y;
    int x = blockIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    if (x >= width) return;

    int tile_y = ty + HALF;

    // Pixel centrale
    if (y < height) {
        int idx = y * pitch + x;
        tile[tile_y] = input[idx];
    } else {
        tile[tile_y] = make_uchar4(0, 0, 0, 0);
    }

    // Halo superiore
    if (ty < HALF) {
        int yy = max(y - HALF, 0);
        int idx = yy * pitch + x;
        tile[ty] = input[idx];
    }

    // Halo inferiore
    if (ty >= BLOCK_SIZE - HALF) {
        int yy = min(y + HALF, height - 1);
        int idx = yy * pitch + x;
        tile[tile_y + HALF] = input[idx];
    }

    __syncthreads();

    if (y >= height) return;

    float r=0, g=0, b=0;

    for (int k = -HALF; k <= HALF; k++) {
        float w = d_kernel_1d[k + HALF];
        uchar4 p = tile[tile_y + k];
        r += p.x * w;
        g += p.y * w;
        b += p.z * w;
    }

    int out = y * pitch + x;
    uchar4 outp = input[out];   // preserva alpha
    outp.x = (unsigned char)r;
    outp.y = (unsigned char)g;
    outp.z = (unsigned char)b;
    output[out] = outp;
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
    unsigned char *d_input, *d_output, *d_temp;
    
    /*float h_kernel_float[KERNEL_SIZE * KERNEL_SIZE];
    for(int i=0;i<KERNEL_SIZE*KERNEL_SIZE;i++)
        h_kernel_float[i] = (float)h_kernel[i];*/
    
    // Alloca memoria GPU
    HANDLE_ERROR(cudaMalloc(&d_input, bytes));
    HANDLE_ERROR(cudaMalloc(&d_output, bytes));
    HANDLE_ERROR(cudaMalloc(&d_temp, bytes));

    HANDLE_ERROR(cudaMemcpy(d_input, src->pixels, bytes, cudaMemcpyHostToDevice));
    
    // Copia il kernel nella memoria costante della GPU
    HANDLE_ERROR(cudaMemcpyToSymbol(d_kernel_1d, h_kernel_1d, KERNEL_SIZE * sizeof(float)));    
    
    //dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 256 thread per blocco
    //dim3 grid((width + BLOCK_SIZE -1)/BLOCK_SIZE, (height + BLOCK_SIZE -1)/BLOCK_SIZE);
    
    dim3 blockH(BLOCK_SIZE, 1);
    dim3 gridH((width + BLOCK_SIZE - 1) / BLOCK_SIZE, height);

    dim3 blockV(1, BLOCK_SIZE);
    dim3 gridV(width, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int pitch4 = pitch/4;
    //gaussian_blur_kernel_shared<<<grid, block>>>(d_input, d_output, width, height, pitch);
    gaussian_horizontal_shared<<<gridH, blockH>>>(reinterpret_cast<uchar4*>(d_input), reinterpret_cast<uchar4*>(d_temp), width, height, pitch4);
    cudaDeviceSynchronize();
    gaussian_vertical_shared<<<gridV, blockV>>>(reinterpret_cast<uchar4*>(d_temp), reinterpret_cast<uchar4*>(d_output), width, height, pitch4);
    checkCUDAError("Errore nella chiamata al kernel 'gaussian_blur_kernel()'");

    HANDLE_ERROR(cudaMemcpy(dst->pixels, d_output, bytes, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(d_input));
    HANDLE_ERROR(cudaFree(d_output));
    HANDLE_ERROR(cudaFree(d_temp));

    SDL_UnlockSurface(src);
    SDL_UnlockSurface(dst);

    return dst;
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

    SDL_Surface* image = load_img("images/img-gaussianblur.jpg");
    
    
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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Execution time: %.3f ms\n", ms-4000);

    return 0;
}

