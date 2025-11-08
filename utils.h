#ifndef UTILS_H
#define UTILS_H

#ifndef __AMX_TILE__
#define __AMX_TILE__ 0
#endif

#ifdef __CUDACC__
#define __builtin_ia32_ldtilecfg(x) ((void)0)
#define __builtin_ia32_sttilecfg(x) ((void)0)
#define __builtin_ia32_tileloadd64(x, y, z) ((void)0)
#define __builtin_ia32_tilestored64(x, y, z) ((void)0)
#define __builtin_ia32_tilezero(x) ((void)0)
#endif

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