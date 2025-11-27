# Percorsi delle directory
EDGEDIR=edge-detection
GBDIR=gaussian-blur
RGBDIR=rgb-to-grayscale
UTILS=utils.cu

# File sorgenti
EDGESERIAL=$(EDGEDIR)/serial.cu
GBSERIAL=$(GBDIR)/serial.cu
GBNPARALLEL=$(GBDIR)/naive-parallel.cu
GBOPARALLEL=$(GBDIR)/optimized-parallel.cu
RGBSERIAL=$(RGBDIR)/serial.cu
RGBNPARALLEL=$(RGBDIR)/naive-parallel.cu

# Binari prodotti
EDGE_SERIAL_BIN=$(EDGEDIR)/serial
RGB_SERIAL_BIN=$(RGBDIR)/serial
RGB_NPARALLEL_BIN=$(RGBDIR)/naive-parallel
GB_SERIAL_BIN=$(GBDIR)/serial
GB_NPARALLEL_BIN=$(GBDIR)/naive-parallel
GB_OPARALLEL_BIN=$(GBDIR)/optimized-parallel
# Flags compilazione
NVCC=nvcc
NVCC_FLAGS=`sdl2-config --cflags --libs` -lSDL2_image -arch=compute_89 -code=sm_89 # MODIFY THIS LAST TWO FLAGS WITH YOUR COMPUTE CAPABILITY


all: $(EDGE_SERIAL_BIN) $(RGB_SERIAL_BIN) $(RGB_NPARALLEL_BIN) $(GB_SERIAL_BIN) $(GB_NPARALLEL_BIN) $(GB_OPARALLEL_BIN)

$(EDGE_SERIAL_BIN): $(EDGESERIAL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(RGB_SERIAL_BIN): $(RGBSERIAL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(RGB_NPARALLEL_BIN): $(RGBNPARALLEL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(GB_SERIAL_BIN): $(GBSERIAL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(GB_NPARALLEL_BIN): $(GBNPARALLEL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(GB_OPARALLEL_BIN): $(GBOPARALLEL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

.PHONY: all clean

clean:
	rm -f $(EDGE_SERIAL_BIN) $(RGB_SERIAL_BIN) $(RGB_NPARALLEL_BIN) $(GB_SERIAL_BIN) $(GB_NPARALLEL_BIN) $(GB_OPARALLEL_BIN)