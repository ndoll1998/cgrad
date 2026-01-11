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
CFLAGS  := -I$(OPENBLAS_PREFIX)/include -Iinclude -O3
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
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# --------- Test Rules ---------
TEST_BIN_SRCS := $(shell find $(TESTS_DIR) -type f -name 'test_*.c')
TEST_BINS := $(patsubst $(TESTS_DIR)/%.c,$(BUILD_TESTS_DIR)/%,$(TEST_BIN_SRCS))

BENCH_BIN := $(BUILD_TESTS_DIR)/bench_contiguous

test: $(OBJ_NO_MAIN) $(TEST_OBJ_FILES)
	@FILE_VAL="$(FILE)"; \
	if [ -z "$$FILE_VAL" ]; then \
		FILE_VAL="tests/test_all.c"; \
	fi; \
	FILE_PATH=$$(echo $$FILE_VAL | sed 's|^\./||'); \
	BIN_PATH=$$(echo $$FILE_PATH | sed 's|^$(TESTS_DIR)/||;s|\.c$$||'); \
	$(MAKE) $(BUILD_TESTS_DIR)/$$BIN_PATH; \
	echo "Running $(BUILD_TESTS_DIR)/$$BIN_PATH"; \
	$(BUILD_TESTS_DIR)/$$BIN_PATH

bench: build $(BENCH_BIN)
	@echo "Running benchmark: $(BENCH_BIN)"
	@$(BENCH_BIN)

# Pattern rule for test object files (arbitrary nesting)
$(BUILD_TESTS_DIR)/%.o: $(TESTS_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -I$(CMOCKA_PREFIX)/include -I$(OPENBLAS_PREFIX)/include -Iinclude -c $< -o $@

# Test binaries
# Helper variable: all object files except main.o
OBJ_NO_MAIN := $(filter-out $(OBJ_DIR)/main.o,$(OBJ_FILES))

$(BUILD_TESTS_DIR)/%: $(BUILD_TESTS_DIR)/%.o $(OBJ_NO_MAIN)
	$(CC) $^ $(CMOCKA_PREFIX)/lib/libcmocka.a -o $@ $(LDFLAGS)

$(BUILD_TESTS_DIR)/bench_contiguous: $(BUILD_TESTS_DIR)/bench_contiguous.o $(OBJ_NO_MAIN)
	$(CC) $^ -o $@ $(LDFLAGS)


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