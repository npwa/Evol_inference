# Cross-compile for Linux/aarch64 from x86 (tier T2). Static, so the binary runs under qemu-user
# (via binfmt or `qemu-aarch64 -cpu <model> ./qdot_test`) without an aarch64 sysroot.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_EXE_LINKER_FLAGS "-static")
