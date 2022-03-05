# NVIDIA C++ compiler and flags.
NVCC        = nvcc

LDFLAGS	= -L$(CUDA_ROOT)/lib64 -lcudart

# Hardware-specific flags for NVIDIA GPU generations. Add any/all.
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM35    := -gencode arch=compute_37,code=sm_37
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SM52    := -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM75    := -gencode arch=compute_75,code=\"sm_75,compute_75\"

GENCODE_FLAGS   := $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM37)             \
$(GENCODE_SM50) $(GENCODE_SM52) $(GENCODE_SM60) $(GENCODE_SM70) $(GENCODE_SM75)

#name executable
EXE	        = SPSO
OBJ	        = main.o

#build up flags and targets
NVCCFLAGS=-O3 $(GENCODE_FLAGS) -Xcompiler -march=native

NVCCFLAGS+= -c --std=c++03

NVCCFLAGS+= -DWITH_CUBLAS -I $(CUDA_ROOT)/include
LDFLAGS+= -lcublas -L $(CUDA_ROOT)/lib64 -Xcompiler \"-Wl,-rpath,$(CUDA_ROOT)/lib64\"

all: $(EXE)

main.o: main.cu kernel.cu
	$(NVCC) -c -o $@ main.cu $(NVCCFLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) $(LDFLAGS) -o $(EXE)

clean:
	/bin/rm -rf *.o $(EXE)
