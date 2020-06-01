#include <stdint.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include <omp.h>
#include <stdlib.h>
#include <vector>
#include <future>
#include <math.h>
#include<CL/cl.hpp>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define TEST_COUNT 1
#define SUB_MATRIX_SIZE 65
// #define SUB_MATRIX_SIZE 2

typedef std::chrono::high_resolution_clock Clock;

class Random
{
public:
    static size_t seed;
    inline static size_t next()
    {
        seed = ((seed * 1103515245) + 12345) & 0x7fffffff;
        return seed;
    }
};
size_t Random::seed = 1000;

class Matrix
{
public:
    size_t cols, rows;
    double *data;

public:
    Matrix(size_t rows, size_t cols)
    {
        this->rows = rows;
        this->cols = cols;
        data = new double[cols * rows]();
    }

    ~Matrix()
    {
        delete[] data;
    }

    Matrix &operator=(const Matrix &src)
    {
        this->rows = src.rows;
        this->cols = src.cols;
        this->data = new double[cols * rows];
        std::copy(src.data, src.data + (cols * rows), this->data);
        return *this;
    }

    size_t total_size()
    {
        return cols * rows;
    }

    void randomize_data()
    {
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < cols; j++)
            {
                size_t id = i * cols + j;
                data[id] = (id * Random::next()) % 10;
            }
        }
    }

    inline const double &get(size_t row, size_t col) const
    {
        assert(row < rows && col < cols);
        return data[row * cols + col];
    }

    inline double &get_mut(size_t row, size_t col)
    {
        assert(row < rows && col < cols);
        return data[row * cols + col];
    }

    void print() const
    {
        const int padding = 6;
        std::stringstream padding_stream;

        for (size_t j = 0; j < cols; j++)
        {
            padding_stream << "---";
            for (size_t i = 0; i < padding; i++)
            {
                padding_stream << "-";
            }
        }
        padding_stream << "-";
        std::string line_sep;
        padding_stream >> line_sep;

        std::cout << line_sep << std::endl;

        for (size_t i = 0; i < rows; i++)
        {
            std::cout << "| ";
            for (size_t j = 0; j < cols; j++)
            {
                size_t id = i * cols + j;
                std::stringstream ss;
                ss << data[id];
                std::string ret;
                ss >> ret;
                for (size_t k = 0; k < padding - ret.length(); k++)
                {
                    std::cout << " ";
                }
                std::cout << ret << " | ";
            }
            std::cout << std::endl;
            std::cout << line_sep << std::endl;
        }
    }

    static Matrix mult_mat(const Matrix &A, const Matrix &B)
    {
        Matrix result = Matrix(A.rows, B.cols);
        double sum;
        for (size_t i = 0; i < A.rows; ++i)
        {
            for (size_t j = 0; j < B.cols; ++j)
            {
                sum = 0.0;
                for (size_t k = 0; k < A.cols; ++k)
                {
                    sum += A.get(i, k) * B.get(k, j);
                }
                result.get_mut(i, j) = sum;
            }
        }

        return result;
    }
};

int main() 
{
    cl_int err;
    size_t rows = 4;
    size_t cols = 4;
    size_t cols2 = 4;

    Matrix A(rows, cols);
    A.randomize_data();
    Matrix B(cols, cols2);
    B.randomize_data();

    Matrix C(rows, cols2);

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    auto device = devices.front();
    std::ifstream clFile("matmul.cl");
    std::string src(std::istreambuf_iterator<char>(clFile), (std::istreambuf_iterator<char>()));

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

    cl::Context context(device);
    cl::Program program(context, sources);

    err = program.build("-cl-std=CL1.2");
    cl::CommandQueue queue(context, device);

    cl::Buffer bufA(context, CL_MEM_READ_ONLY, A.total_size()*sizeof(double), NULL, NULL);
    cl::Buffer bufB(context, CL_MEM_READ_ONLY, B.total_size()*sizeof(double), NULL, NULL);
    cl::Buffer bufC(context, CL_MEM_READ_WRITE, C.total_size()*sizeof(double), NULL, NULL);

    err = queue.enqueueWriteBuffer(bufA, CL_TRUE, 0, A.total_size()*sizeof(double), A.data);
    err = queue.enqueueWriteBuffer(bufB, CL_TRUE, 0, B.total_size()*sizeof(double), B.data);
    err = queue.enqueueWriteBuffer(bufC, CL_TRUE, 0, C.total_size()*sizeof(double), C.data);


    // cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buf));
    cl::Kernel kernel(program, "parallel_mult_mat", &err);
    kernel.setArg<unsigned long>(0, rows);
    kernel.setArg<unsigned long>(1, cols2);
    kernel.setArg<unsigned long>(2, cols);
    kernel.setArg(3, bufA);
    kernel.setArg(4, bufB);
    kernel.setArg(5, bufC);

    cl::NDRange local(min(rows, 32), min(cols2, 32));
    cl::NDRange global(rows, cols2);

    std::vector<cl::Event> events;

    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &events, NULL);
    err = cl::WaitForEvents(events);
    err = queue.enqueueReadBuffer(bufC, CL_TRUE, 0, C.total_size()*sizeof(double), C.data, NULL, NULL);

    std::cout << "A:" << std::endl;
    A.print();
    std::cout << "B:" << std::endl;
    B.print();
    std::cout << "C:" << std::endl;
    C.print();
    std::cout << "AxB:" << std::endl;
    Matrix::mult_mat(A, B).print();
}