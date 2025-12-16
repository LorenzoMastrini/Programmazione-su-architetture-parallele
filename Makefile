# Percorsi delle directory
RGBDIR=rgb-to-grayscale
EDGEDIR=edge-detection
GBDIR=gaussian-blur
UTILS=utils.cu
INFO=info.cu

# File sorgenti
RGB_SERIAL=$(RGBDIR)/serial.cu
RGB_NAIVE_PARALLEL=$(RGBDIR)/naive-parallel.cu
RGB_OPTIMIZED_PARALLEL=$(RGBDIR)/optimized-parallel.cu
EDGESERIAL=$(EDGEDIR)/serial.cu
GBSERIAL=$(GBDIR)/serial.cu
GBNPARALLEL=$(GBDIR)/naive-parallel.cu
GBOPARALLEL=$(GBDIR)/optimized-parallel.cu

# Binari prodotti
RGB_SERIAL_BIN=$(RGBDIR)/serial
RGB_NAIVE_PARALLEL_BIN=$(RGBDIR)/naive-parallel
RGB_OPTIMIZED_PARALLEL_BIN=$(RGBDIR)/optimized-parallel
EDGE_SERIAL_BIN=$(EDGEDIR)/serial
GB_SERIAL_BIN=$(GBDIR)/serial
GB_NPARALLEL_BIN=$(GBDIR)/naive-parallel
GB_OPARALLEL_BIN=$(GBDIR)/optimized-parallel
INFO_BIN=info

# Flags compilazione
NVCC=nvcc
NVCC_FLAGS=`sdl2-config --cflags --libs` -lSDL2_image -arch=compute_89 -code=sm_89 # MODIFY THIS LAST TWO FLAGS WITH YOUR COMPUTE CAPABILITY


all: $(INFO_BIN) $(RGB_SERIAL_BIN) $(RGB_NAIVE_PARALLEL_BIN) $(RGB_OPTIMIZED_PARALLEL_BIN) $(EDGE_SERIAL_BIN) $(GB_SERIAL_BIN) $(GB_NPARALLEL_BIN) $(GB_OPARALLEL_BIN)

# Info
$(INFO_BIN): $(INFO) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

# RGB to grayscale
$(RGB_SERIAL_BIN): $(RGB_SERIAL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(RGB_NAIVE_PARALLEL_BIN): $(RGB_NAIVE_PARALLEL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(RGB_OPTIMIZED_PARALLEL_BIN): $(RGB_OPTIMIZED_PARALLEL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

# Edge Detection
$(EDGE_SERIAL_BIN): $(EDGESERIAL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(GB_SERIAL_BIN): $(GBSERIAL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(GB_NPARALLEL_BIN): $(GBNPARALLEL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(GB_OPARALLEL_BIN): $(GBOPARALLEL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

.PHONY: all clean

clean:
	rm -f $(INFO_BIN) $(RGB_SERIAL_BIN) $(RGB_NAIVE_PARALLEL_BIN) $(EDGE_SERIAL_BIN) $(GB_SERIAL_BIN) $(GB_NPARALLEL_BIN) $(GB_OPARALLEL_BIN)