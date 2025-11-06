#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <cuda_runtime.h>

// Macro to handle CUDA errors
#define HANDLE_ERROR(err) HandleError(err, __FILE__, __LINE__)

void HandleError(cudaError_t err, const char* file, int line);
void checkCUDAError(const char *msg);
int init_SDL();
SDL_Surface* load_img(const char* path);
SDL_Window* create_window(const char* title);

#endif