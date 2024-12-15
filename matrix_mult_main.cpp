#include "omp.h"
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <string>
#define MKL
#ifdef MKL
#include "mkl.h"
#endif

using namespace std;

void generation(double* mat, size_t size)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> uniform_distance(-2.001, 2.001);
    for (size_t i = 0; i < size * size; i++)
        mat[i] = uniform_distance(gen);
}

//Total time: 0.205 sec
//Total time mkl : 0.007 sec

//void matrix_mult(double* a, double* b, double* res, size_t size)
//{
//#pragma omp parallel for
//    for (int i = 0; i < size; i++)
//    {
//        for (int j = 0; j < size; j++)
//        {
//            for (int k = 0; k < size; k++)
//            {
//                res[i * size + j] += a[i * size + k] * b[k * size + j];
//            }
//        }
//    }
//}

//Total time : 0.017 sec
//Total time mkl : 0.007 sec


//intrinsic
void matrix_mult(double* a, double* b, double* res, size_t size) {
    int block_size = 32; // Размер блока для блочного умножения

#pragma omp parallel for
    // Проход по блокам
    for (int i = 0; i < size; i += block_size) {
        for (int j = 0; j < size; j += block_size) {
            for (int k = 0; k < size; k += block_size) {
                // Обработка подматриц
                for (int ii = i; ii < i + block_size && ii < size; ++ii) {
                    for (int kk = k; kk < k + block_size && kk < size; ++kk) {
                        // Загрузка элемента
                        double tmp = a[ii * size + kk];

                        int min_jj = std::min(j + block_size, static_cast<int>(size)); //гарантируем, что не выйдем за пределы матрицы

                        // Обработка вектора из 4-х элементов для результата
                        for (int jj = j; jj < min_jj; jj += 4) {
                            // Загружаем 4 значения из b
                            __m256d b_vec = _mm256_loadu_pd(&b[kk * size + jj]);

                            // Определяем вектор для результата
                            __m256d res_vec = _mm256_loadu_pd(&res[ii * size + jj]);

                            // Умножаем и добавляем
                            res_vec = _mm256_fmadd_pd(_mm256_set1_pd(tmp), b_vec, res_vec); // a * b + c

                            // Сохраняем результат
                            _mm256_storeu_pd(&res[ii * size + jj], res_vec);
                        }
                    }
                }
            }
        }
    }
}


int main()
{
    double* mat, * mat_mkl, * a, * b, * a_mkl, * b_mkl;
    size_t size = 1000;


    chrono::time_point<chrono::system_clock> start, end;

    mat = new double[size * size];
    a = new double[size * size];
    b = new double[size * size];

    generation(a, size);
    generation(b, size);
    memset(mat, 0, size * size * sizeof(double));



#ifdef MKL
    mat_mkl = new double[size * size];
    a_mkl = new double[size * size];
    b_mkl = new double[size * size];
    memcpy(a_mkl, a, sizeof(double) * size * size);
    memcpy(b_mkl, b, sizeof(double) * size * size);
    memset(mat_mkl, 0, size * size * sizeof(double));
#endif

    start = chrono::system_clock::now();
    matrix_mult(a, b, mat, size);
    end = chrono::system_clock::now();


    int elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
        (end - start).count();
    cout << "Total time: " << elapsed_seconds / 1000.0 << " sec" << endl;

#ifdef MKL
    start = chrono::system_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, 1.0, a_mkl, size, b_mkl, size, 0.0, mat_mkl, size);
    end = chrono::system_clock::now();

    elapsed_seconds = chrono::duration_cast<chrono::milliseconds>
        (end - start).count();
    cout << "Total time mkl: " << elapsed_seconds / 1000.0 << " sec" << endl;

    int flag = 0;
    for (unsigned int i = 0; i < size * size; i++)
        if (abs(mat[i] - mat_mkl[i]) > size * 1e-14) {
            flag = 1;
        }
    if (flag)
        cout << "fail" << endl;
    else
        cout << "correct" << endl;

    delete (a_mkl);
    delete (b_mkl);
    delete (mat_mkl);
#endif

    delete (a);
    delete (b);
    delete (mat);

    //system("pause");

    return 0;
}
