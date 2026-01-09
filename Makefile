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
# CMocka Dependency
# -----------------------------
CMOCKA_VERSION := 2.0.1
CMOCKA_NAME    := cmocka-$(CMOCKA_VERSION)
CMOCKA_URL     := https://cmocka.org/files/2.0/$(CMOCKA_NAME).tar.xz
CMOCKA_TARBALL := /tmp/$(CMOCKA_NAME).tar.xz
CMOCKA_TMPDIR  := /tmp/cmocka_build
CMOCKA_SRCDIR  := $(CMOCKA_TMPDIR)/$(CMOCKA_NAME)
CMOCKA_PREFIX  ?= $(CURDIR)/deps/cmocka

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
.PHONY: install_openblas install_criterion install_cmocka clean_openblas clean_criterion clean_cmocka build clean test

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

# Install CMocka
install_cmocka:
	@echo "==> Installing CMocka $(CMOCKA_VERSION) (static library only)"
	@rm -rf $(CMOCKA_TMPDIR)
	@mkdir -p $(CMOCKA_TMPDIR)
	@curl -L -o $(CMOCKA_TARBALL) $(CMOCKA_URL)
	@tar -xJf $(CMOCKA_TARBALL) -C $(CMOCKA_TMPDIR)
	@cd $(CMOCKA_SRCDIR) && cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$(CMOCKA_PREFIX) -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF
	@cmake --build $(CMOCKA_SRCDIR)/build
	@cmake --install $(CMOCKA_SRCDIR)/build

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
	mkdir -p build/tests/backends

# Test target
test: build/tests/test_all
	build/tests/test_all

build/tests/bench_contiguous.o: tests/bench_contiguous.c include/cgrad_tensor.h
	mkdir -p build/tests
	$(CC) $(CFLAGS) -c tests/bench_contiguous.c -o build/tests/bench_contiguous.o

build/tests/bench_contiguous: build/tests/bench_contiguous.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) $(CFLAGS) build/tests/bench_contiguous.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o -o build/tests/bench_contiguous $(LDFLAGS)

# CMocka test for cgrad_tensor_f32_cpu
build/tests/backends/test_cgrad_tensor_f32_cpu.o: tests/backends/test_cgrad_tensor_f32_cpu.c include/backends/cgrad_tensor_f32_cpu.h
	mkdir -p build/tests/backends
	$(CC) -I$(CMOCKA_PREFIX)/include -I$(OPENBLAS_PREFIX)/include -Iinclude -c tests/backends/test_cgrad_tensor_f32_cpu.c -o build/tests/backends/test_cgrad_tensor_f32_cpu.o

build/tests/backends/test_cgrad_tensor_f32_cpu: build/tests/backends/test_cgrad_tensor_f32_cpu.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) build/tests/backends/test_cgrad_tensor_f32_cpu.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o $(CMOCKA_PREFIX)/lib/libcmocka.a -o build/tests/backends/test_cgrad_tensor_f32_cpu $(LDFLAGS)

bench: build/tests/bench_contiguous
	build/tests/bench_contiguous

# CMocka test for cgrad_layout
# Standalone test object files (with main)
build/tests/test_cgrad_layout.o: tests/test_cgrad_layout.c include/cgrad_layout.h
	mkdir -p build/tests
	$(CC) -I$(CMOCKA_PREFIX)/include -Iinclude -c tests/test_cgrad_layout.c -o build/tests/test_cgrad_layout.o

build/tests/test_cgrad_layout: build/tests/test_cgrad_layout.o build/cgrad_layout.o
	$(CC) build/tests/test_cgrad_layout.o build/cgrad_layout.o $(CMOCKA_PREFIX)/lib/libcmocka.a -o build/tests/test_cgrad_layout

build/tests/test_cgrad_tensor.o: tests/test_cgrad_tensor.c include/cgrad_tensor.h
	mkdir -p build/tests
	$(CC) -I$(CMOCKA_PREFIX)/include -I$(OPENBLAS_PREFIX)/include -Iinclude -c tests/test_cgrad_tensor.c -o build/tests/test_cgrad_tensor.o

build/tests/test_cgrad_tensor: build/tests/test_cgrad_tensor.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) build/tests/test_cgrad_tensor.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o $(CMOCKA_PREFIX)/lib/libcmocka.a -o build/tests/test_cgrad_tensor $(LDFLAGS)

build/tests/backends/test_cgrad_tensor_f32_cpu.o: tests/backends/test_cgrad_tensor_f32_cpu.c include/backends/cgrad_tensor_f32_cpu.h
	mkdir -p build/tests/backends
	$(CC) -I$(CMOCKA_PREFIX)/include -I$(OPENBLAS_PREFIX)/include -Iinclude -c tests/backends/test_cgrad_tensor_f32_cpu.c -o build/tests/backends/test_cgrad_tensor_f32_cpu.o

build/tests/backends/test_cgrad_tensor_f32_cpu: build/tests/backends/test_cgrad_tensor_f32_cpu.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) build/tests/backends/test_cgrad_tensor_f32_cpu.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o $(CMOCKA_PREFIX)/lib/libcmocka.a -o build/tests/backends/test_cgrad_tensor_f32_cpu $(LDFLAGS)

# Unified test binary (include all test sources directly)
build/tests/test_all.o: tests/test_all.c
	mkdir -p build/tests
	$(CC) -I$(CMOCKA_PREFIX)/include -I$(OPENBLAS_PREFIX)/include -Iinclude -c tests/test_all.c -o build/tests/test_all.o

build/tests/test_all: build/tests/test_all.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) build/tests/test_all.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o $(CMOCKA_PREFIX)/lib/libcmocka.a -o build/tests/test_all $(LDFLAGS)

# CMocka test for cgrad_tensor
build/tests/test_cgrad_tensor.o: tests/test_cgrad_tensor.c include/cgrad_tensor.h
	mkdir -p build/tests
	$(CC) -I$(CMOCKA_PREFIX)/include -I$(OPENBLAS_PREFIX)/include -Iinclude -c tests/test_cgrad_tensor.c -o build/tests/test_cgrad_tensor.o

build/tests/test_cgrad_tensor: build/tests/test_cgrad_tensor.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) build/tests/test_cgrad_tensor.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o $(CMOCKA_PREFIX)/lib/libcmocka.a -o build/tests/test_cgrad_tensor $(LDFLAGS)

# Unified test binary (optional, for CI)
build/tests/test_all.o: tests/test_all.c
	mkdir -p build/tests
	$(CC) -I$(CMOCKA_PREFIX)/include -I$(OPENBLAS_PREFIX)/include -Iinclude -c tests/test_all.c -o build/tests/test_all.o

build/tests/test_all: build/tests/test_all.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	$(CC) build/tests/test_all.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o $(CMOCKA_PREFIX)/lib/libcmocka.a -o build/tests/test_all $(LDFLAGS)

# Cleanup
clean: clean_openblas clean_criterion clean_cmocka
	@echo "==> Cleaning project"
	@rm -f $(OUT)
	@rm -rf build

clean_openblas:
	@rm -rf $(OPENBLAS_TMPDIR)
	@rm -f $(OPENBLAS_TARBALL)

clean_criterion:
	@rm -rf $(CRITERION_TMPDIR)
	@rm -f $(CRITERION_TARBALL)

clean_cmocka:
	@rm -rf $(CMOCKA_TMPDIR)
	@rm -f $(CMOCKA_TARBALL)
build/tests/backends/test_cgrad_tensor_f32_cpu: build/tests/backends/test_cgrad_tensor_f32_cpu.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
build/tests/backends/test_cgrad_tensor_f32_cpu: build/tests/backends/test_cgrad_tensor_f32_cpu.o build/cgrad_tensor.o build/cgrad_layout.o build/cgrad_backend.o build/cgrad_tensor_f32_cpu.o
	@rm -rf $(OPENBLAS_TMPDIR)
