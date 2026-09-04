# SOC Make

CC       = gcc
CFLAGS   = -O3 -std=c11 -fPIC -Wall -Wextra -pedantic -I$(INC_DIR)
LDFLAGS  = -shared -lm
SRC_DIR  = ./src
INC_DIR  = ./include
BUILD_DIR = ./build
BIN_OUT	 = libsoc.so

SOURCES  = $(SRC_DIR)/shape.c \
           $(SRC_DIR)/operators.c \
           $(SRC_DIR)/metrics.c \
           $(SRC_DIR)/generators.c

HEADERS  = $(INC_DIR)/types.h \
           $(INC_DIR)/shape.h \
           $(INC_DIR)/operators.h \
           $(INC_DIR)/metrics.h \
           $(INC_DIR)/generators.h

TARGET   = $(BUILD_DIR)/$(BIN_OUT)

# Default target
all: $(TARGET)

# Shared object build
$(TARGET): $(SOURCES) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SOURCES)

# Clean
clean:
	rm -rf $(BUILD_DIR)

# build + run Python test scripts (if any)
test: $(TARGET)
	python -m pytest tests/ -v

.PHONY: all clean test
