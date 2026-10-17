cmake_minimum_required(VERSION 3.18)
project(ptb_praat_browser LANGUAGES C CXX)
if(NOT EMSCRIPTEN)
  message(FATAL_ERROR "This target requires the isolated Emscripten toolchain")
endif()
if(NOT PARSELMOUTH_SOURCE)
  message(FATAL_ERROR "Set PARSELMOUTH_SOURCE to the verified 0.4.7 sdist")
endif()
add_subdirectory(${PARSELMOUTH_SOURCE}/extern/fmt fmt)
add_subdirectory(${PARSELMOUTH_SOURCE}/praat praat)
# No application threads: a dedicated browser Worker owns this whole module.
get_target_property(PRAAT_BROWSER_FLAGS praat COMPILE_OPTIONS)
list(REMOVE_ITEM PRAAT_BROWSER_FLAGS -pthread)
set_property(TARGET praat PROPERTY COMPILE_OPTIONS ${PRAAT_BROWSER_FLAGS})
get_target_property(PRAAT_BROWSER_PUBLIC_FLAGS praat INTERFACE_COMPILE_OPTIONS)
list(REMOVE_ITEM PRAAT_BROWSER_PUBLIC_FLAGS -pthread)
set_property(TARGET praat PROPERTY INTERFACE_COMPILE_OPTIONS ${PRAAT_BROWSER_PUBLIC_FLAGS})
add_executable(ptb_praat praat.cpp)
target_link_libraries(ptb_praat PRIVATE praat)
target_compile_options(ptb_praat PRIVATE -fexceptions -ffp-contract=off)
target_compile_options(praat PRIVATE -fexceptions -ffp-contract=off)
set_target_properties(ptb_praat PROPERTIES SUFFIX .js CXX_STANDARD 17)
target_link_options(ptb_praat PRIVATE
  -fexceptions -sDISABLE_EXCEPTION_CATCHING=0 -sALLOW_MEMORY_GROWTH=1
  -sMAXIMUM_MEMORY=1073741824 -sINITIAL_MEMORY=134217728 -sSTACK_SIZE=1048576
  -sMODULARIZE=1 -sEXPORT_NAME=createPtbPraat -sENVIRONMENT=worker,node
  -sFILESYSTEM=1 -sNO_EXIT_RUNTIME=1 -sDYNAMIC_EXECUTION=0
  "-sEXPORTED_FUNCTIONS=['_malloc','_free','_ptb_spectrogram','_ptb_pitch','_ptb_values','_ptb_grid','_ptb_error']"
  "-sEXPORTED_RUNTIME_METHODS=['HEAPF64','HEAPU8','UTF8ToString']")
