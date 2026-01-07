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
CFLAGS  := -I$(OPENBLAS_PREFIX)/include
LDFLAGS := -L$(OPENBLAS_PREFIX)/lib -lopenblas
SRC     := src/main.c
OUT     := main

# -----------------------------
# Targets
# -----------------------------
.PHONY: install_openblas clean_openblas build clean

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

# Compile src/main.c
build:
	@echo "==> Compiling $(SRC)"
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

# Cleanup
clean: clean_openblas
	@echo "==> Cleaning project"
	@rm -f $(OUT)

clean_openblas:
	@rm -rf $(OPENBLAS_TMPDIR)
	@rm -f $(OPENBLAS_TARBALL)

