# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/sapphie/.clion/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/sapphie/.clion/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sapphie/CLionProjects/DigitRecogniser

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sapphie/CLionProjects/DigitRecogniser/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/DigitRecogniser.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DigitRecogniser.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DigitRecogniser.dir/flags.make

CMakeFiles/DigitRecogniser.dir/main.cpp.o: CMakeFiles/DigitRecogniser.dir/flags.make
CMakeFiles/DigitRecogniser.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sapphie/CLionProjects/DigitRecogniser/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DigitRecogniser.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/DigitRecogniser.dir/main.cpp.o -c /home/sapphie/CLionProjects/DigitRecogniser/main.cpp

CMakeFiles/DigitRecogniser.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DigitRecogniser.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sapphie/CLionProjects/DigitRecogniser/main.cpp > CMakeFiles/DigitRecogniser.dir/main.cpp.i

CMakeFiles/DigitRecogniser.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DigitRecogniser.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sapphie/CLionProjects/DigitRecogniser/main.cpp -o CMakeFiles/DigitRecogniser.dir/main.cpp.s

CMakeFiles/DigitRecogniser.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/DigitRecogniser.dir/main.cpp.o.requires

CMakeFiles/DigitRecogniser.dir/main.cpp.o.provides: CMakeFiles/DigitRecogniser.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/DigitRecogniser.dir/build.make CMakeFiles/DigitRecogniser.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/DigitRecogniser.dir/main.cpp.o.provides

CMakeFiles/DigitRecogniser.dir/main.cpp.o.provides.build: CMakeFiles/DigitRecogniser.dir/main.cpp.o


# Object files for target DigitRecogniser
DigitRecogniser_OBJECTS = \
"CMakeFiles/DigitRecogniser.dir/main.cpp.o"

# External object files for target DigitRecogniser
DigitRecogniser_EXTERNAL_OBJECTS =

DigitRecogniser: CMakeFiles/DigitRecogniser.dir/main.cpp.o
DigitRecogniser: CMakeFiles/DigitRecogniser.dir/build.make
DigitRecogniser: CMakeFiles/DigitRecogniser.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sapphie/CLionProjects/DigitRecogniser/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable DigitRecogniser"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DigitRecogniser.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DigitRecogniser.dir/build: DigitRecogniser

.PHONY : CMakeFiles/DigitRecogniser.dir/build

CMakeFiles/DigitRecogniser.dir/requires: CMakeFiles/DigitRecogniser.dir/main.cpp.o.requires

.PHONY : CMakeFiles/DigitRecogniser.dir/requires

CMakeFiles/DigitRecogniser.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DigitRecogniser.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DigitRecogniser.dir/clean

CMakeFiles/DigitRecogniser.dir/depend:
	cd /home/sapphie/CLionProjects/DigitRecogniser/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sapphie/CLionProjects/DigitRecogniser /home/sapphie/CLionProjects/DigitRecogniser /home/sapphie/CLionProjects/DigitRecogniser/cmake-build-debug /home/sapphie/CLionProjects/DigitRecogniser/cmake-build-debug /home/sapphie/CLionProjects/DigitRecogniser/cmake-build-debug/CMakeFiles/DigitRecogniser.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DigitRecogniser.dir/depend

