CXX      := g++
CXXFLAGS := -std=c++17 -g -O0 -Iinclude -Iinclude/flatbuffers -Iinclude/lite
LDFLAGS  := -L/usr/lib/aarch64-linux-gnu -ledgetpu -lpthread -ldl -lm

LIBS := \
  lib/libtensorflow-lite.a \
  lib/libruy.a \
  lib/libXNNPACK.a \
  lib/libfft2d_fftsg.a \
  lib/libfft2d_fftsg2d.a \
  lib/libflatbuffers.a \
  lib/libfarmhash.a \
  lib/libcpuinfo.a \
  lib/libclog.a \
  lib/libpthreadpool.a \
  lib/libabsl_synchronization.a \
  lib/libabsl_graphcycles_internal.a \
  lib/libabsl_flags_internal.a \
  lib/libabsl_flags_config.a \
  lib/libabsl_flags_marshalling.a \
  lib/libabsl_flags_program_name.a \
  lib/libabsl_flags.a \
  lib/libabsl_flags_registry.a \
  lib/libabsl_int128.a \
  lib/libabsl_status.a \
  lib/libabsl_base.a \
  lib/libabsl_throw_delegate.a \
  lib/libabsl_dynamic_annotations.a \
  lib/libabsl_malloc_internal.a \
  lib/libabsl_log_severity.a \
  lib/libabsl_raw_logging_internal.a \
  lib/libabsl_spinlock_wait.a \
  lib/libabsl_time_zone.a \
  lib/libabsl_civil_time.a \
  lib/libabsl_time.a \
  lib/libabsl_bad_variant_access.a \
  lib/libabsl_bad_optional_access.a \
  lib/libabsl_strings.a \
  lib/libabsl_cord.a \
  lib/libabsl_strings_internal.a \
  lib/libabsl_str_format_internal.a \
  lib/libabsl_demangle_internal.a \
  lib/libabsl_symbolize.a \
  lib/libabsl_stacktrace.a \
  lib/libabsl_debugging_internal.a \
  lib/libabsl_hash.a \
  lib/libabsl_city.a

SRC      := src/inference.cpp
TARGET   := build/inference

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ $(LIBS) $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./build/inference