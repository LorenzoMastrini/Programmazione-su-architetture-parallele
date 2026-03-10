#include "../utils.cu"
#include <stdio.h>

#define KERNEL_SIZE 5
#define BLOCK_SIZE 16
#define HALF 2
#define NUM_STREAMS 16  // Numero di stream da utilizzare

//__constant__ float d_kernel[KERNEL_SIZE*KERNEL_SIZE];
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

__global__ void gaussian_horizontal(unsigned char *input, unsigned char *output, int width, int height, int pitch){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float r = 0, g = 0, b = 0;

    for (int k = -HALF; k <= HALF; k++) {
        int px = min(max(x + k, 0), width - 1);
        int idx = y * pitch + px * 4;

        float w = d_kernel_1d[k + HALF];
        r += input[idx + 0] * w;
        g += input[idx + 1] * w;
        b += input[idx + 2] * w;
    }

    int out = y * pitch + x * 4;
    output[out + 0] = (unsigned char)r;
    output[out + 1] = (unsigned char)g;
    output[out + 2] = (unsigned char)b;
    output[out + 3] = input[out + 3];
}

__global__ void gaussian_vertical(unsigned char *input, unsigned char *output, int width, int height, int pitch){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float r = 0, g = 0, b = 0;

    for (int k = -HALF; k <= HALF; k++) {
        int py = min(max(y + k, 0), height - 1);
        int idx = py * pitch + x * 4;

        float w = d_kernel_1d[k + HALF];
        r += input[idx + 0] * w;
        g += input[idx + 1] * w;
        b += input[idx + 2] * w;
    }

    int out = y * pitch + x * 4;
    output[out + 0] = (unsigned char)r;
    output[out + 1] = (unsigned char)g;
    output[out + 2] = (unsigned char)b;
    output[out + 3] = input[out + 3];
}

// Versioni dei kernel con offset per multi-stream
__global__ void gaussian_horizontal_offset(unsigned char *input, unsigned char *output, int width, int tile_height, int pitch, int start_y, int total_height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;  // y locale al tile
    int y = start_y + y_local;  // y globale nell'immagine

    if (x >= width || y_local >= tile_height) return;

    float r = 0, g = 0, b = 0;

    for (int k = -HALF; k <= HALF; k++) {
        int px = min(max(x + k, 0), width - 1);
        int idx = y * pitch + px * 4;

        float w = d_kernel_1d[k + HALF];
        r += input[idx + 0] * w;
        g += input[idx + 1] * w;
        b += input[idx + 2] * w;
    }

    int out = y * pitch + x * 4;
    output[out + 0] = (unsigned char)r;
    output[out + 1] = (unsigned char)g;
    output[out + 2] = (unsigned char)b;
    output[out + 3] = input[out + 3];
}

__global__ void gaussian_vertical_offset(unsigned char *input, unsigned char *output, int width, int tile_height, int pitch, int start_y, int total_height){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y_local = blockIdx.y * blockDim.y + threadIdx.y;  // y locale al tile
    int y = start_y + y_local;  // y globale nell'immagine

    if (x >= width || y_local >= tile_height) return;

    float r = 0, g = 0, b = 0;

    for (int k = -HALF; k <= HALF; k++) {
        int py_global = y + k;
        int py = min(max(py_global, 0), total_height - 1);  // Clamp rispetto all'intera immagine
        int idx = py * pitch + x * 4;

        float w = d_kernel_1d[k + HALF];
        r += input[idx + 0] * w;
        g += input[idx + 1] * w;
        b += input[idx + 2] * w;
    }

    int out = y * pitch + x * 4;
    output[out + 0] = (unsigned char)r;
    output[out + 1] = (unsigned char)g;
    output[out + 2] = (unsigned char)b;
    output[out + 3] = input[out + 3];
}


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

// Funzione di Gaussian blur per immagine RGB con multi-stream
// L'immagine viene divisa in NUM_STREAMS tile orizzontali, ognuno processato su uno stream separato.
// Questo permette di sovrapporre i trasferimenti H2D/D2D con l'esecuzione dei kernel.
// Per modificare il numero di stream, cambia la macro NUM_STREAMS (valori tipici: 2-8).
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
    
    // Registra la memoria host come pinned
    HANDLE_ERROR(cudaHostRegister(src->pixels, bytes, cudaHostRegisterDefault));
    HANDLE_ERROR(cudaHostRegister(dst->pixels, bytes, cudaHostRegisterDefault));
    
    // Copia il kernel nella memoria costante della GPU (deve essere sincrono)
    HANDLE_ERROR(cudaMemcpyToSymbol(d_kernel_1d, h_kernel_1d, KERNEL_SIZE * sizeof(float)));
    
    // Alloca memoria GPU globale (condivisa tra tutti gli stream)
    unsigned char *d_input, *d_output, *d_temp;
    HANDLE_ERROR(cudaMalloc(&d_input, bytes));
    HANDLE_ERROR(cudaMalloc(&d_output, bytes));
    HANDLE_ERROR(cudaMalloc(&d_temp, bytes));
    
    // Crea array di stream
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        HANDLE_ERROR(cudaStreamCreate(&streams[i]));
    }
    
    // Calcola le dimensioni dei tile
    int tile_height = (height + NUM_STREAMS - 1) / NUM_STREAMS;  // Altezza approssimativa per tile
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    
    // Processa ogni tile su uno stream diverso
    for (int s = 0; s < NUM_STREAMS; s++) {
        int start_y = s * tile_height;
        int end_y = min((s + 1) * tile_height, height);
        int tile_h = end_y - start_y;
        
        if (tile_h <= 0) continue;
        
        // Offset per i puntatori (in byte)
        size_t offset = start_y * pitch;
        size_t tile_bytes = tile_h * pitch;
        
        // Copia asincrona del tile da host a device su questo stream
        HANDLE_ERROR(cudaMemcpyAsync(d_input + offset, (unsigned char*)src->pixels + offset, tile_bytes, cudaMemcpyHostToDevice, streams[s]));
        
        // Calcola la griglia per questo tile
        dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (tile_h + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // Esegui kernel horizontal sul tile con offset (dopo la copia)
        gaussian_horizontal_offset<<<grid, block, 0, streams[s]>>>(d_input, d_temp, width, tile_h, pitch, start_y, height);
        
        // Esegui kernel vertical sul tile con offset (dipende dal risultato di horizontal)
        gaussian_vertical_offset<<<grid, block, 0, streams[s]>>>(d_temp, d_output, width, tile_h, pitch, start_y, height);
        
        // Copia asincrona del risultato del tile da device a host
        HANDLE_ERROR(cudaMemcpyAsync((unsigned char*)dst->pixels + offset, d_output + offset, tile_bytes, cudaMemcpyDeviceToHost, streams[s]));
    }
    
    // Sincronizza tutti gli stream
    for (int i = 0; i < NUM_STREAMS; i++) {
        HANDLE_ERROR(cudaStreamSynchronize(streams[i]));
    }
    
    // Verifica errori CUDA
    checkCUDAError("Errore nella chiamata ai kernel gaussiani");
    
    // Distruggi tutti gli stream
    for (int i = 0; i < NUM_STREAMS; i++) {
        HANDLE_ERROR(cudaStreamDestroy(streams[i]));
    }
    
    // Pulizia
    HANDLE_ERROR(cudaHostUnregister(src->pixels));
    HANDLE_ERROR(cudaHostUnregister(dst->pixels));
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

