# =============================
# cgrad Makefile (Simplified)
# =============================

# --------- Dependency Versions ---------
OPENBLAS_VERSION := 0.3.30
CMOCKA_VERSION   := 2.0.1
BENCHMARK_VERSION := 1.9.4

# --------- Dependency Paths/URLs ---------
# --------- Dependency Paths/URLs ---------
OPENBLAS_NAME    := OpenBLAS-$(OPENBLAS_VERSION)
OPENBLAS_URL     := https://github.com/OpenMathLib/OpenBLAS/releases/download/v$(OPENBLAS_VERSION)/$(OPENBLAS_NAME).tar.gz
OPENBLAS_TARBALL := /tmp/$(OPENBLAS_NAME).tar.gz
OPENBLAS_TMPDIR  := /tmp/openblas_build
OPENBLAS_SRCDIR  := $(OPENBLAS_TMPDIR)/$(OPENBLAS_NAME)
OPENBLAS_PREFIX  ?= $(CURDIR)/deps/OpenBLAS

CMOCKA_NAME      := cmocka-$(CMOCKA_VERSION)
CMOCKA_URL       := https://cmocka.org/files/2.0/$(CMOCKA_NAME).tar.xz
CMOCKA_TARBALL   := /tmp/$(CMOCKA_NAME).tar.xz
CMOCKA_TMPDIR    := /tmp/cmocka_build
CMOCKA_SRCDIR    := $(CMOCKA_TMPDIR)/$(CMOCKA_NAME)
CMOCKA_PREFIX    ?= $(CURDIR)/deps/cmocka

BENCHMARK_NAME      := benchmark-$(BENCHMARK_VERSION)
BENCHMARK_URL       := https://github.com/google/benchmark/archive/refs/tags/v$(BENCHMARK_VERSION).tar.gz
BENCHMARK_TARBALL   := /tmp/$(BENCHMARK_NAME).tar.gz
BENCHMARK_TMPDIR    := /tmp/benchmark_build
BENCHMARK_SRCDIR    := $(BENCHMARK_TMPDIR)/$(BENCHMARK_NAME)
BENCHMARK_PREFIX    ?= $(CURDIR)/deps/benchmark

# --------- Compiler/Flags ---------
CC      := gcc
CFLAGS  := -I$(OPENBLAS_PREFIX)/include -I$(CMOCKA_PREFIX)/include -Iinclude -O3
LDFLAGS := -L$(OPENBLAS_PREFIX)/lib -lopenblas

# --------- Project Structure ---------
SRC_DIR      := src
INCLUDE_DIR  := include
OBJ_DIR      := build
TESTS_DIR    := tests
BUILD_TESTS_DIR   := $(OBJ_DIR)/tests

SRC_FILES := $(shell find $(SRC_DIR) -name '*.c')
OBJ_FILES := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))

OUT := main

# --------- Test Files ---------
TEST_SRC_FILES := $(shell find $(TESTS_DIR) -name '*.c')
TEST_OBJ_FILES := $(patsubst $(TESTS_DIR)/%.c,$(BUILD_TESTS_DIR)/%.o,$(TEST_SRC_FILES))

# --------- Phony Targets ---------
.PHONY: all build test clean install_deps install_openblas install_cmocka install_benchmark clean_openblas clean_cmocka clean_benchmark

# --------- Default Target ---------
all: build

# --------- Dependency Installation ---------
install_deps: install_openblas install_cmocka install_benchmark

install_openblas:
	@echo "==> Installing OpenBLAS $(OPENBLAS_VERSION)"
	@rm -rf $(OPENBLAS_TMPDIR)
	@mkdir -p $(OPENBLAS_TMPDIR)
	@curl -L -o $(OPENBLAS_TARBALL) $(OPENBLAS_URL)
	@tar -xzf $(OPENBLAS_TARBALL) -C $(OPENBLAS_TMPDIR)
	@cd $(OPENBLAS_SRCDIR) && make && make install PREFIX=$(OPENBLAS_PREFIX)

install_cmocka:
	@echo "==> Installing CMocka $(CMOCKA_VERSION) (static library only)"
	@rm -rf $(CMOCKA_TMPDIR)
	@mkdir -p $(CMOCKA_TMPDIR)
	@curl -L -o $(CMOCKA_TARBALL) $(CMOCKA_URL)
	@tar -xJf $(CMOCKA_TARBALL) -C $(CMOCKA_TMPDIR)
	@cd $(CMOCKA_SRCDIR) && cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$(CMOCKA_PREFIX) -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=OFF
	@cmake --build $(CMOCKA_SRCDIR)/build
	@cmake --install $(CMOCKA_SRCDIR)/build

install_benchmark:
	@echo "==> Installing Google Benchmark $(BENCHMARK_VERSION)"
	@rm -rf $(BENCHMARK_TMPDIR)
	@mkdir -p $(BENCHMARK_TMPDIR)
	@curl -L -o $(BENCHMARK_TARBALL) $(BENCHMARK_URL)
	@tar -xzf $(BENCHMARK_TARBALL) -C $(BENCHMARK_TMPDIR)
	@cd $(BENCHMARK_SRCDIR) && cmake -E make_directory build
	@cd $(BENCHMARK_SRCDIR) && cmake -E chdir build cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=on -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(BENCHMARK_PREFIX) ..
	@cmake --build $(BENCHMARK_SRCDIR)/build --config Release
	@cmake --install $(BENCHMARK_SRCDIR)/build --prefix $(BENCHMARK_PREFIX)

clean_benchmark:
	@rm -rf $(BENCHMARK_TMPDIR)
	@rm -f $(BENCHMARK_TARBALL)

# --------- Build Rules ---------
build: $(OUT)

$(OUT): $(OBJ_FILES)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# --------- Test Rules ---------
TEST_BIN_SRCS := $(shell find $(TESTS_DIR) -type f -name 'test_*.c')
TEST_BINS := $(patsubst $(TESTS_DIR)/%.c,$(BUILD_TESTS_DIR)/%,$(TEST_BIN_SRCS))

BENCHMARKS_DIR := benchmarks
BENCHMARKS_BUILD_DIR := build/benchmarks
BENCHMARKS := $(BENCHMARKS_BUILD_DIR)/bench_contiguous

CXX := g++
BENCHMARK_CXXFLAGS := -std=c++11 -I$(BENCHMARK_PREFIX)/include -Iinclude -I$(OPENBLAS_PREFIX)/include
BENCHMARK_LDFLAGS := -L$(BENCHMARK_PREFIX)/lib -lbenchmark -lpthread -L$(OPENBLAS_PREFIX)/lib -lopenblas

test: $(OBJ_NO_MAIN) $(TEST_OBJ_FILES)
	$(MAKE) clean;
	@FILE_VAL="$(FILE)"; \
	if [ -z "$$FILE_VAL" ]; then \
		FILE_VAL="tests/test_all.c"; \
	fi; \
	FILE_PATH=$$(echo $$FILE_VAL | sed 's|^\./||'); \
	BIN_PATH=$$(echo $$FILE_PATH | sed 's|^$(TESTS_DIR)/||;s|\.c$$||'); \
	$(MAKE) $(BUILD_TESTS_DIR)/$$BIN_PATH; \
	echo "Running $(BUILD_TESTS_DIR)/$$BIN_PATH"; \
	$(BUILD_TESTS_DIR)/$$BIN_PATH

bench: $(BENCHMARKS)
	@echo "Running benchmark: $(BENCHMARKS)"
	@$(BENCHMARKS)

$(BENCHMARKS_BUILD_DIR)/%.o: $(BENCHMARKS_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(BENCHMARK_CXXFLAGS) -c $< -o $@

$(BENCHMARKS_BUILD_DIR)/%: $(BENCHMARKS_BUILD_DIR)/%.o
	$(CXX) $^ $(filter-out $(OBJ_DIR)/main.o,$(OBJ_FILES)) -o $@ $(BENCHMARK_LDFLAGS)

# Test binaries
# Helper variable: all object files except main.o
OBJ_NO_MAIN := $(filter-out $(OBJ_DIR)/main.o,$(OBJ_FILES))

$(BUILD_TESTS_DIR)/%.o: $(TESTS_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_TESTS_DIR)/%: $(BUILD_TESTS_DIR)/%.o $(OBJ_NO_MAIN)
	$(CC) $^ $(CMOCKA_PREFIX)/lib/libcmocka.a -o $@ $(LDFLAGS)


# --------- Clean Rules ---------
clean: clean_openblas clean_cmocka
	@echo "==> Cleaning project"
	@rm -f $(OUT)
	@rm -rf $(OBJ_DIR)

clean_openblas:
	@rm -rf $(OPENBLAS_TMPDIR)
	@rm -f $(OPENBLAS_TARBALL)

clean_cmocka:
	@rm -rf $(CMOCKA_TMPDIR)
	@rm -f $(CMOCKA_TARBALL)
	@rm -f $(CMOCKA_TARBALL)
