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

#define max(a, b) ((a > b) ? (a) : (b))
#define min(a, b) ((a < b) ? (a) : (b))

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

private:
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

private:
    inline static void _parallel_mult_mat_square(const Matrix &A, const Matrix &B, Matrix &result, 
        size_t dim, size_t i_start, size_t j_start, size_t k_start)
    {
        size_t i_end = min(i_start + dim, B.cols);
        size_t j_end = min(j_start + dim, B.rows);
        size_t k_end = min(k_start + dim, A.rows);
        for (size_t i = i_start; i < i_end; i++)
        {
            for (size_t j = j_start; j < j_end; j++)
            {
                double B_val = B.get(j, i);
                for (size_t k = k_start; k < k_end; k++)
                {
                    result.get_mut(k, i) += A.get(k, j) * B_val;
                }
            }
        }
    } 

public:
    static Matrix parallel_mult_mat(const Matrix &A, const Matrix &B)
    {
        Matrix result = Matrix(A.rows, B.cols);

        size_t row_boxes = ceil((float) result.rows/SUB_MATRIX_SIZE);
        size_t col_boxes = ceil((float) result.cols/SUB_MATRIX_SIZE);
        #pragma omp parallel for
        for(size_t x = 0; x < (row_boxes * col_boxes); x++)
        {
            size_t i = x % col_boxes;
            size_t k = x / col_boxes;
            for(size_t j = 0; j < A.cols; j += SUB_MATRIX_SIZE)
            {
                _parallel_mult_mat_square(A, B, result, SUB_MATRIX_SIZE, i * SUB_MATRIX_SIZE, j, k * SUB_MATRIX_SIZE);
            }
        }

        return result;
    }
};

class TestConfig
{
public:
    size_t nrows;
    size_t ncols;
    size_t ncols2;
    size_t test_count;

    TestConfig(size_t nrows, size_t ncols, size_t ncols2, size_t test_count)
        : nrows(nrows), ncols(ncols), ncols2(ncols2), test_count(test_count)
    { }
};

size_t run_test(TestConfig config)
{
    
    Matrix A = Matrix(config.nrows, config.ncols);
    A.randomize_data();
    Matrix B = Matrix(config.ncols, config.ncols2);
    B.randomize_data();

    size_t min_time = -1;
    for (size_t i = 0; i < config.test_count; i++)
    {
        // we need cols threads, each will go through and do the mul for a column, once a column is complete, it will go back and do the second mul, and so on
        auto t1 = Clock::now();
        Matrix result = Matrix::parallel_mult_mat(A, B);
        auto t2 = Clock::now();
        size_t t = (t2 - t1).count();
        min_time = min(min_time, t);
    }

    return min_time;
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cerr << "usage: matmult nrows ncols ncols2 thread_count outfile" << std::endl;
        return EXIT_FAILURE;
    }

    size_t nrows, ncols, ncols2, thread_count;
    nrows = atoi(argv[1]);
    ncols = atoi(argv[2]);
    ncols2 = atoi(argv[3]);
    thread_count = atoi(argv[4]);

    std::ofstream result_file;
    result_file.open(argv[5]);

    // TestConfig config = TestConfig(nrows, ncols, ncols2, 1);
    // long time = run_test(config);
    // std::cout << "time=" << time << std::endl;

    Matrix A = Matrix(nrows, ncols);
    A.randomize_data();
    Matrix B = Matrix(ncols, ncols2);
    B.randomize_data();

    auto t1 = Clock::now();
    Matrix::mult_mat(A, B).print();
    auto t2 = Clock::now();
    size_t t = (t2 - t1).count();

    t1 = Clock::now();
    Matrix::parallel_mult_mat(A, B).print();
    t2 = Clock::now();

    std::cout << "time1=" << t << std::endl;
    std::cout << "time2=" << (t2 - t1).count() << std::endl;

    result_file.close();

    return EXIT_SUCCESS;
}