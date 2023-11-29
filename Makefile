# Simple Makefile to handle release/debug modes
# As well as other CMake command line args
# (which are hard to type and remember)

all: build 

build:
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=RELEASE ..
	$(MAKE) -C ./build

debug:
	mkdir -p debug 
	cd debug && cmake -DCMAKE_BUILD_TYPE=DEBUG ..
	$(MAKE) -C ./debug

test: build 
	./build/bin/unittests

test_debug: debug
	./debug/bin/unittests

clean:
	rm -rf build 
	rm -rf debug 
	rm -rf externals
