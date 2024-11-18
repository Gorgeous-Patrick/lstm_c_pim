CFLAGS = -Wall -Wextra -std=c2x -g
CC = gcc  

SOURCE_DIR = ./src
BUILD_DIR = ./build

C_EXT = c

C_SOURCES = $(wildcard $(SOURCE_DIR)/*.$(C_EXT))
C_OBJECTS = $(patsubst $(SOURCE_DIR)/%.$(C_EXT), $(BUILD_DIR)/%.o, $(C_SOURCES))

#define build types
PROD_FLAGS := -O3 -march=native -flto -DNDEBUG

# set build variable to dev by default
BUILD ?= dev

# set build flags based on build variable
ifeq ($(BUILD),prod)
	CFLAGS += $(PROD_FLAGS)
endif


COMPILE = $(CC) $(CFLAGS) -I./include

PROJECT=main
all: $(PROJECT)

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.$(C_EXT)
	$(COMPILE) -c $< -o $@

$(PROJECT): $(C_OBJECTS)
	$(COMPILE) $(C_OBJECTS) -o $(PROJECT)

clean:
	rm -rf $(PROJECT) $(C_OBJECTS)