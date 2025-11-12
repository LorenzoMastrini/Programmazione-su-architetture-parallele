# Percorsi delle directory
EDGEDIR=edge-detection
GBDIR=gaussian-blur
RGBDIR=rgb-to-grayscale
UTILS=utils.cu

# File sorgenti
EDGESERIAL=$(EDGEDIR)/serial.cu
GBSERIAL=$(GBDIR)/serial.cu
GBPARALLEL=$(GBDIR)/naive-parallel.cu
RGBSERIAL=$(RGBDIR)/serial.cu
RGBPARALLEL=$(RGBDIR)/parallel.cu

# Binari prodotti
EDGE_SERIAL_BIN=$(EDGEDIR)/serial
RGB_SERIAL_BIN=$(RGBDIR)/serial
RGB_PARALLEL_BIN=$(RGBDIR)/parallel
GB_SERIAL_BIN=$(GBDIR)/serial
GB_PARALLEL_BIN=$(GBDIR)/naive-parallel

# Flags compilazione
NVCC=nvcc
NVCC_FLAGS=-Xcompiler -march=x86-64 -Xcompiler -mno-amx-tile -Xcompiler -D__AMX_TILE__=0 `sdl2-config --cflags --libs` -lSDL2_image


all: $(EDGE_SERIAL_BIN) $(RGB_SERIAL_BIN) $(RGB_PARALLEL_BIN) $(GB_SERIAL_BIN) $(GB_PARALLEL_BIN)

$(EDGE_SERIAL_BIN): $(EDGESERIAL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(RGB_SERIAL_BIN): $(RGBSERIAL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(RGB_PARALLEL_BIN): $(RGBPARALLEL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(GB_SERIAL_BIN): $(GBSERIAL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

$(GB_PARALLEL_BIN): $(GBPARALLEL) $(UTILS)
	$(NVCC) $^ -o $@ $(NVCC_FLAGS)

.PHONY: all clean

clean:
	rm -f $(EDGE_SERIAL_BIN) $(RGB_SERIAL_BIN) $(RGB_PARALLEL_BIN) $(GB_SERIAL_BIN) $(GB_PARALLEL_BIN)