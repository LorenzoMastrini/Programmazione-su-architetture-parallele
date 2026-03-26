#include "../utils.h"



// Conversione di un'immagine da RGB a scala di grigi
void rgb_to_grayscale(uchar4 *pixels, unsigned char *output, int number_of_pixels) {
    for(int i = 0; i < number_of_pixels; i++) {
        // Access each pixel
        uchar4 p = pixels[i];

        float r = p.x; // Red
        float g = p.y; // Green
        float b = p.z; // Blue

        // Grayscale formula (based on non-linear human perception of color)
        unsigned char gray = 0.3 * r + 0.59 * g + 0.11 * b;
        output[i] = gray;
    }
}



int main() {
    init_SDL();

    SDL_Surface *image = load_img("images/img.jpg");
    SDL_LockSurface(image); // Lock to access pixels
    const int width = image->w;
    const int height = image->h;
    const int number_of_pixels = width * height;

    uchar4 *pixels = (uchar4*) image->pixels;
    
    // Output image
    unsigned char* grayscale_image = (unsigned char*) malloc(number_of_pixels* sizeof(unsigned char));


    // Profiling
    cudaEvent_t start, stop;
    float time;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    rgb_to_grayscale(pixels, grayscale_image, number_of_pixels);

    // Profiling
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Time: %.2f ms\n", time);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    

    show_image(grayscale_image, width, height, "Conversione da RGB a scala di grigi - Versione seriale");

    // ----- Clean up -----
    SDL_UnlockSurface(image);
    SDL_FreeSurface(image);
    free(grayscale_image);
    IMG_Quit();
    SDL_Quit();

    return 0;
}