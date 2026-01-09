# -----------------------------
# OpenBLAS Dependency
# -----------------------------
OPENBLAS_VERSION := 0.3.30
OPENBLAS_NAME    := OpenBLAS-$(OPENBLAS_VERSION)
OPENBLAS_URL     := https://github.com/OpenMathLib/OpenBLAS/releases/download/v$(OPENBLAS_VERSION)/$(OPENBLAS_NAME).tar.gz
OPENBLAS_TARBALL := /tmp/$(OPENBLAS_NAME).tar.gz
OPENBLAS_TMPDIR  := /tmp/openblas_build
OPENBLAS_SRCDIR  := $(OPENBLAS_TMPDIR)/$(OPENBLAS_NAME)
OPENBLAS_PREFIX  ?= $(CURDIR)/deps/OpenBLAS

# -----------------------------
# Compiler / Project
# -----------------------------
CC      := gcc
CFLAGS  := -I$(OPENBLAS_PREFIX)/include -Iinclude
LDFLAGS := -L$(OPENBLAS_PREFIX)/lib -lopenblas
SRC     := src/main.c src/cgrad_tensor.c src/cgrad_layout.c src/cgrad_backend.c src/backends/cgrad_tensor_f32_cpu.c
OBJ     := build/main.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
OUT     := main

# -----------------------------
# Targets
# -----------------------------
.PHONY: install_openblas clean_openblas build clean test

# Install OpenBLAS
install_openblas:
	@echo "==> Installing OpenBLAS $(OPENBLAS_VERSION)"
	@rm -rf $(OPENBLAS_TMPDIR)
	@mkdir -p $(OPENBLAS_TMPDIR)
	@curl -L -o $(OPENBLAS_TARBALL) $(OPENBLAS_URL)
	@tar -xzf $(OPENBLAS_TARBALL) -C $(OPENBLAS_TMPDIR)
	@cd $(OPENBLAS_SRCDIR) && \
		make && \
		make install PREFIX=$(OPENBLAS_PREFIX)

# Build object files and link
build: $(OUT)

build/main.o: src/main.c include/cgrad_tensor.h
	mkdir -p build
	$(CC) $(CFLAGS) -c src/main.c -o build/main.o

build/cgrad_tensor.o: src/cgrad_tensor.c include/cgrad_tensor.h
	mkdir -p build
	$(CC) $(CFLAGS) -c src/cgrad_tensor.c -o build/cgrad_tensor.o

build/cgrad_layout.o: src/cgrad_layout.c include/cgrad_layout.h
	mkdir -p build
	$(CC) $(CFLAGS) -c src/cgrad_layout.c -o build/cgrad_layout.o

build/cgrad_backend.o: src/cgrad_backend.c include/cgrad_backend.h
	mkdir -p build
	$(CC) $(CFLAGS) -c src/cgrad_backend.c -o build/cgrad_backend.o

build/cgrad_tensor_f32_cpu.o: src/backends/cgrad_tensor_f32_cpu.c include/backends/cgrad_tensor_f32_cpu.h
	mkdir -p build
	$(CC) $(CFLAGS) -c src/backends/cgrad_tensor_f32_cpu.c -o build/cgrad_tensor_f32_cpu.o

$(OUT): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(OUT) $(LDFLAGS)

build:
	mkdir -p build
	mkdir -p build/tests
# Test target
test: build/tests/test_gemm build/tests/test_transpose build/tests/test_contiguous
	build/tests/test_gemm
	build/tests/test_transpose
	build/tests/test_contiguous

build/tests/test_contiguous.o: tests/test_contiguous.c include/cgrad_tensor.h
	mkdir -p build/tests
	$(CC) $(CFLAGS) -c tests/test_contiguous.c -o build/tests/test_contiguous.o

build/tests/test_contiguous: build/tests/test_contiguous.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) $(CFLAGS) build/tests/test_contiguous.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o -o build/tests/test_contiguous $(LDFLAGS)

build/tests/test_gemm.o: tests/test_gemm.c include/cgrad_tensor.h
	mkdir -p build/tests
	$(CC) $(CFLAGS) -c tests/test_gemm.c -o build/tests/test_gemm.o

build/tests/test_gemm: build/tests/test_gemm.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) $(CFLAGS) build/tests/test_gemm.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o -o build/tests/test_gemm $(LDFLAGS)

build/tests/test_transpose.o: tests/test_transpose.c include/cgrad_tensor.h
	mkdir -p build/tests
	$(CC) $(CFLAGS) -c tests/test_transpose.c -o build/tests/test_transpose.o

build/tests/test_transpose: build/tests/test_transpose.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) $(CFLAGS) build/tests/test_transpose.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o -o build/tests/test_transpose $(LDFLAGS)

build/tests/bench_contiguous.o: tests/bench_contiguous.c include/cgrad_tensor.h
	mkdir -p build/tests
	$(CC) $(CFLAGS) -c tests/bench_contiguous.c -o build/tests/bench_contiguous.o

build/tests/bench_contiguous: build/tests/bench_contiguous.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) $(CFLAGS) build/tests/bench_contiguous.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o -o build/tests/bench_contiguous $(LDFLAGS)

bench: build/tests/bench_contiguous
	build/tests/bench_contiguous

# Cleanup
clean: clean_openblas
	@echo "==> Cleaning project"
	@rm -f $(OUT)
	@rm -rf build

clean_openblas:
	@rm -rf $(OPENBLAS_TMPDIR)
	@rm -f $(OPENBLAS_TARBALL)

