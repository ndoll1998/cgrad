# =============================
# cgrad Makefile (Simplified)
# =============================

# --------- Dependency Versions ---------
OPENBLAS_VERSION := 0.3.30
CMOCKA_VERSION   := 2.0.1

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

# --------- Compiler/Flags ---------
CC      := gcc
CFLAGS  := -I$(OPENBLAS_PREFIX)/include -Iinclude
LDFLAGS := -L$(OPENBLAS_PREFIX)/lib -lopenblas

# --------- Project Structure ---------
SRC_DIR      := src
BACKEND_DIR  := $(SRC_DIR)/backends
INCLUDE_DIR  := include
OBJ_DIR      := build
TESTS_DIR    := tests
TESTS_BACKEND_DIR := $(TESTS_DIR)/backends
BUILD_TESTS_DIR   := $(OBJ_DIR)/tests
BUILD_TESTS_BACKEND_DIR := $(BUILD_TESTS_DIR)/backends

SRC_FILES := $(SRC_DIR)/main.c \
             $(SRC_DIR)/cgrad_tensor.c \
             $(SRC_DIR)/cgrad_layout.c \
             $(SRC_DIR)/cgrad_backend.c \
             $(BACKEND_DIR)/cgrad_tensor_f32_cpu.c

OBJ_FILES := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(filter $(SRC_DIR)/%.c,$(SRC_FILES))) \
             $(patsubst $(BACKEND_DIR)/%.c,$(OBJ_DIR)/backends/%.o,$(filter $(BACKEND_DIR)/%.c,$(SRC_FILES)))

OUT := main

# --------- Test Files ---------
TEST_SOURCES := $(wildcard $(TESTS_DIR)/*.c)
TEST_BACKEND_SOURCES := $(wildcard $(TESTS_BACKEND_DIR)/*.c)

# --------- Phony Targets ---------
.PHONY: all build test clean install_deps install_openblas install_cmocka clean_openblas clean_cmocka

# --------- Default Target ---------
all: build

# --------- Dependency Installation ---------
install_deps: install_openblas install_cmocka

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

# --------- Build Rules ---------
build: $(OUT)

$(OUT): $(OBJ_FILES)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/backends/%.o: $(BACKEND_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# --------- Test Rules ---------
TEST_BINS := \
	$(BUILD_TESTS_DIR)/test_all \
	$(BUILD_TESTS_DIR)/test_cgrad_tensor \
	$(BUILD_TESTS_DIR)/test_cgrad_layout \
	$(BUILD_TESTS_BACKEND_DIR)/test_cgrad_tensor_f32_cpu

BENCH_BIN := $(BUILD_TESTS_DIR)/bench_contiguous

test: 
	@mkdir -p $(BUILD_TESTS_BACKEND_DIR)
	$(MAKE) build
	$(MAKE) $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "Running $$t"; $$t; done

bench: build $(BENCH_BIN)
	@echo "Running benchmark: $(BENCH_BIN)"
	@$(BENCH_BIN)

# Pattern rule for test object files
$(BUILD_TESTS_DIR)/%.o: $(TESTS_DIR)/%.c
	@mkdir -p $(BUILD_TESTS_DIR)
	$(CC) -I$(CMOCKA_PREFIX)/include -I$(OPENBLAS_PREFIX)/include -Iinclude -c $< -o $@

$(BUILD_TESTS_BACKEND_DIR)/%.o: $(TESTS_BACKEND_DIR)/%.c | $(BUILD_TESTS_BACKEND_DIR)
	$(CC) -I$(CMOCKA_PREFIX)/include -I$(OPENBLAS_PREFIX)/include -Iinclude -c $< -o $@

$(BUILD_TESTS_BACKEND_DIR):
	mkdir -p $(BUILD_TESTS_BACKEND_DIR)

# Test binaries
# Helper variable: all object files except main.o
OBJ_NO_MAIN := $(filter-out $(OBJ_DIR)/main.o,$(OBJ_FILES))

$(BUILD_TESTS_DIR)/test_all: $(BUILD_TESTS_DIR)/test_all.o $(OBJ_NO_MAIN)
	$(CC) $^ $(CMOCKA_PREFIX)/lib/libcmocka.a -o $@ $(LDFLAGS)

$(BUILD_TESTS_DIR)/test_cgrad_tensor: $(BUILD_TESTS_DIR)/test_cgrad_tensor.o $(OBJ_NO_MAIN)
	$(CC) $^ $(CMOCKA_PREFIX)/lib/libcmocka.a -o $@ $(LDFLAGS)

$(BUILD_TESTS_DIR)/test_cgrad_layout: $(BUILD_TESTS_DIR)/test_cgrad_layout.o $(OBJ_DIR)/cgrad_layout.o
	$(CC) $^ $(CMOCKA_PREFIX)/lib/libcmocka.a -o $@ $(LDFLAGS)

$(BUILD_TESTS_DIR)/bench_contiguous: $(BUILD_TESTS_DIR)/bench_contiguous.o $(OBJ_NO_MAIN)
	$(CC) $^ -o $@ $(LDFLAGS)

$(BUILD_TESTS_BACKEND_DIR)/test_cgrad_tensor_f32_cpu: $(BUILD_TESTS_BACKEND_DIR)/test_cgrad_tensor_f32_cpu.o $(OBJ_NO_MAIN)
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
