NVCC = nvcc
FLAGS = 
cu = ${wildcard .${HOME}/src/*.cu}
obj = ${cu:.cu=.o}

all: ${obj}
	${NVCC} ${FLAGS} main.cu ${obj} -o build/main 

%.o: %.cu
	@echo "compiling $< to $@"
	${NVCC} ${FLAGS} -c $< -o $@