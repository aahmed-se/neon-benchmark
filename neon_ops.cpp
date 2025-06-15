#ifndef NEON_OPERATIONS_H
#define NEON_OPERATIONS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <optional>

namespace py = pybind11;

class NeonOperations {
private:
    static constexpr size_t CACHE_LINE_SIZE = 64;  // bytes
    static constexpr size_t L1_CACHE_SIZE = 32 * 1024;  // 32KB
    static constexpr size_t OPTIMAL_CHUNK_SIZE = L1_CACHE_SIZE / sizeof(float);
    static constexpr size_t VECTOR_SIZE = 4;  // float32x4_t processes 4 elements at once

    static size_t get_thread_count(size_t size) {
        if (size < 10000) return 1;
        size_t thread_count = static_cast<size_t>(std::thread::hardware_concurrency());
        size_t calculated_threads = size / OPTIMAL_CHUNK_SIZE + 1;
        return (thread_count < calculated_threads) ? thread_count : calculated_threads;
    }

    static void add_chunk(const float* ptr1, const float* ptr2, float* ptr_result, size_t start, size_t end) {
        size_t aligned_end = end - (end - start) % VECTOR_SIZE;
        size_t i = start;

        for (; i < aligned_end; i += VECTOR_SIZE) {
            float32x4_t vec1 = vld1q_f32(ptr1 + i);
            float32x4_t vec2 = vld1q_f32(ptr2 + i);
            float32x4_t sum = vaddq_f32(vec1, vec2);
            vst1q_f32(ptr_result + i, sum);
        }

        for (; i < end; ++i) {
            ptr_result[i] = ptr1[i] + ptr2[i];
        }
    }

    static void multiply_chunk(const float* ptr1, const float* ptr2, float* ptr_result, size_t start, size_t end) {
        size_t aligned_end = end - (end - start) % VECTOR_SIZE;
        size_t i = start;

        for (; i < aligned_end; i += VECTOR_SIZE) {
            float32x4_t vec1 = vld1q_f32(ptr1 + i);
            float32x4_t vec2 = vld1q_f32(ptr2 + i);
            float32x4_t product = vmulq_f32(vec1, vec2);
            vst1q_f32(ptr_result + i, product);
        }

        for (; i < end; ++i) {
            ptr_result[i] = ptr1[i] * ptr2[i];
        }
    }

    static void scale_chunk(const float* ptr, float scalar, float* ptr_result, size_t start, size_t end) {
        float32x4_t scalar_vec = vdupq_n_f32(scalar);
        size_t aligned_end = end - (end - start) % VECTOR_SIZE;
        size_t i = start;

        for (; i < aligned_end; i += VECTOR_SIZE) {
            float32x4_t vec = vld1q_f32(ptr + i);
            float32x4_t scaled = vmulq_f32(vec, scalar_vec);
            vst1q_f32(ptr_result + i, scaled);
        }

        for (; i < end; ++i) {
            ptr_result[i] = ptr[i] * scalar;
        }
    }

public:
    static py::array_t<float> add_arrays(py::array_t<float> input1, py::array_t<float> input2, std::optional<py::array_t<float>> output_opt = std::nullopt) {
        py::buffer_info buf1 = input1.request(), buf2 = input2.request();
        if (buf1.size != buf2.size) {
            throw std::runtime_error("Input arrays must have the same size");
        }

        py::array_t<float> result = output_opt.has_value() && output_opt->size() == buf1.size ? output_opt.value() : py::array_t<float>(buf1.size);
        py::buffer_info buf_result = result.request();

        const float* ptr1 = static_cast<float*>(buf1.ptr);
        const float* ptr2 = static_cast<float*>(buf2.ptr);
        float* ptr_result = static_cast<float*>(buf_result.ptr);

        size_t size = static_cast<size_t>(buf1.size);
        size_t num_threads = get_thread_count(size);

        if (num_threads == 1) {
            add_chunk(ptr1, ptr2, ptr_result, 0, size);
        } else {
            std::vector<std::thread> threads;
            size_t chunk_size = size / num_threads;

            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
                threads.emplace_back(add_chunk, ptr1, ptr2, ptr_result, start, end);
            }

            for (auto& thread : threads) {
                thread.join();
            }
        }

        return result;
    }

    static py::array_t<float> multiply_arrays(py::array_t<float> input1, py::array_t<float> input2, std::optional<py::array_t<float>> output_opt = std::nullopt) {
        py::buffer_info buf1 = input1.request(), buf2 = input2.request();
        if (buf1.size != buf2.size) {
            throw std::runtime_error("Input arrays must have the same size");
        }

        py::array_t<float> result = output_opt.has_value() && output_opt->size() == buf1.size ? output_opt.value() : py::array_t<float>(buf1.size);
        py::buffer_info buf_result = result.request();

        const float* ptr1 = static_cast<float*>(buf1.ptr);
        const float* ptr2 = static_cast<float*>(buf2.ptr);
        float* ptr_result = static_cast<float*>(buf_result.ptr);

        size_t size = static_cast<size_t>(buf1.size);
        size_t num_threads = get_thread_count(size);

        if (num_threads == 1) {
            multiply_chunk(ptr1, ptr2, ptr_result, 0, size);
        } else {
            std::vector<std::thread> threads;
            size_t chunk_size = size / num_threads;

            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
                threads.emplace_back(multiply_chunk, ptr1, ptr2, ptr_result, start, end);
            }

            for (auto& thread : threads) {
                thread.join();
            }
        }

        return result;
    }

    static py::array_t<float> scale_array(py::array_t<float> input, float scalar, std::optional<py::array_t<float>> output_opt = std::nullopt) {
        py::buffer_info buf = input.request();
        py::array_t<float> result = output_opt.has_value() && output_opt->size() == buf.size ? output_opt.value() : py::array_t<float>(buf.size);
        py::buffer_info buf_result = result.request();

        const float* ptr = static_cast<float*>(buf.ptr);
        float* ptr_result = static_cast<float*>(buf_result.ptr);

        size_t size = static_cast<size_t>(buf.size);
        size_t num_threads = get_thread_count(size);

        if (num_threads == 1) {
            scale_chunk(ptr, scalar, ptr_result, 0, size);
        } else {
            std::vector<std::thread> threads;
            size_t chunk_size = size / num_threads;

            for (size_t i = 0; i < num_threads; ++i) {
                size_t start = i * chunk_size;
                size_t end = (i == num_threads - 1) ? size : (i + 1) * chunk_size;
                threads.emplace_back(scale_chunk, ptr, scalar, ptr_result, start, end);
            }

            for (auto& thread : threads) {
                thread.join();
            }
        }

        return result;
    }
};

PYBIND11_MODULE(neon_ops, m) {
    m.doc() = "NEON-accelerated array operations for Apple M-series processors";
    m.def("add_arrays", &NeonOperations::add_arrays, py::arg("input1"), py::arg("input2"), py::arg("output_opt") = std::nullopt, "Add two arrays using NEON");
    m.def("multiply_arrays", &NeonOperations::multiply_arrays, py::arg("input1"), py::arg("input2"), py::arg("output_opt") = std::nullopt, "Multiply two arrays using NEON");
    m.def("scale_array", &NeonOperations::scale_array, py::arg("input"), py::arg("scalar"), py::arg("output_opt") = std::nullopt, "Scale array by a constant using NEON");
}

#endif // NEON_OPERATIONS_H