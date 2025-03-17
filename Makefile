.DEFAULT_GOAL := build-all

#*****************************************#
#*               VARIABLES               *#
#*****************************************#

THIS_MAKEFILE := $(realpath $(lastword $(MAKEFILE_LIST)))
THIS_MAKEFILE_DIR := $(patsubst %/,%,$(dir $(THIS_MAKEFILE)))
REPO_DIR := $(THIS_MAKEFILE_DIR)
REPO_PREFIX := ../$(notdir $(abspath $(REPO_DIR)))
THIRD_PARTY_DIR := $(THIS_MAKEFILE_DIR)/third_party

ifeq ($(CPM_SOURCE_CACHE),)
  CPM_SOURCE_CACHE := $(THIRD_PARTY_DIR)
endif

CMAKE := cmake
MAKE ?= make
TYPE ?= D
BUILD_DIR ?= build
ENVS += CPM_SOURCE_CACHE=$(CPM_SOURCE_CACHE)

SHELL := /usr/bin/env $(ENVS) bash

#*****************************************#
#*               UTILITIES               *#
#*****************************************#
#* Makefile debugging
print-%: ; @$(warning $* is $($*) ($(value $*)) (from $(origin $*)))
define message
@echo -n "make[top]: "
@echo $(1)
endef


#*****************************************#
#*                OPTIONS                *#
#*****************************************#

CMAKE_OPTIONS :=

ifneq ($(TYPE),)
  BUILD_TYPE_R := Release
  BUILD_TYPE_D := Debug
  BUILD_TYPE_RD := RelWithDebInfo
  BUILD_TYPE_MR := MinSizeRel

  CMAKE_OPTIONS += -DCMAKE_BUILD_TYPE=$(BUILD_TYPE_$(TYPE))
endif

ifneq ($(COVERAGE),)
  CODE_COVERAGE_OPTIONS := 0 1
  CMAKE_OPTIONS += -DENABLE_TEST_COVERAGE=$(COVERAGE)
endif

EMPTY :=
ifneq ($(SANITIZER),)
  SANITIZER_OPTIONS := Address Memory MemoryWithOrigins Undefined Thread Leak
  CMAKE_OPTIONS += -DUSE_SANITIZER=$(subst $(empty) $(empty),;,$(SANITIZER))
endif

ifneq ($(STATIC_CHECK),)
  STATIC_CHECK_OPTIONS := clang-tidy iwyu cppcheck
  CMAKE_OPTIONS += $(foreach TYPE,$(STATIC_CHECK),-DUSE_STATIC_ANALYZER=$(TYPE))
endif

ifneq ($(CCACHE),)
  CCACHE_OPTIONS := ON OFF
  CMAKE_OPTIONS += -DUSE_CCACHE=$(CCACHE)
endif


#*****************************************#
#*                ACTIONS                *#
#*****************************************#

# Build the project
build: FORCE
	$(CMAKE) -B build/lib $(CMAKE_OPTIONS)
	$(CMAKE) --build build/lib

source-all: FORCE
	$(call message, Source)
	$(CMAKE) -S cmake/ -B build $(CMAKE_OPTIONS)

build-sample: source-all
	$(call message, Build)
	$(CMAKE) --build build/sample

build-test: source-all
	$(call message, Build)
	$(CMAKE) --build build/tests

build-all: build build-sample build-test

.PHONY: format
format: format-cpp

.PHONY: format-cpp
format-cpp:
	@while IFS= read -r file; do \
        test -z "$${file%%#*}" || clang-format -i -style=file "$$file"; \
    done < $(BUILD_DIR)/compile_files.list

#* Linting
.PHONY: test
test: build-test
	$(call message, Run for test)
	CTEST_OUTPUT_ON_FAILURE=1 $(CMAKE) --build build/tests --target test

# Clean build files
clean:
	@rm -rf $(BUILD_DIR)

# Rebuild the project
rebuild: clean build
.NOTPARALLEL: rebuild

FORCE:
