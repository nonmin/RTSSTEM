NVCC=nvcc
CFLAGS="-arch=sm_30"
LDFLAGS=-lcublas

PDSICORE=src/si.cu src/common.cu src/utils.cu src/wt.cu src/filters.cpp
PDSIOBJ=build/si.o build/common.o build/utils.o build/wt.o build/filters.o

demo: $(PDSICORE) src/main.cpp src/io.cpp
	mkdir -p build
	$(NVCC) -g $(CFLAGS) -odir build -c $^
	$(NVCC) $(CFLAGS) -o RTSSTEM $(PDSIOBJ) build/main.o build/io.o $(LDFLAGS)


libpdwt.so: $(PDSICORE)
	mkdir -p build
	$(NVCC) $(CFLAGS) --ptxas-options=-v --compiler-options '-fPIC' -odir build -c $^
	$(NVCC) $(CFLAGS)  -o $@ --shared $(PDSIOBJ) $(LDFLAGS)

clean:
	rm -rf build
