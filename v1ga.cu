#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include"device_launch_parameters.h"
#include"cuda_runtime.h"
#include"cuda_runtime_api.h"
using namespace std;

int n, m, p, q;
const int szB = 3550, szL = 150;

double *matrix, *kernel, *result;

inline int read() {
    char c = getchar();
    int x = 0, f = 1;
    while (isdigit(c))
        x = (x << 3) + (x << 1) + (c^48), c = getchar();
    return x * f;
}

inline void read(double &r)
{
    double x=0,t=0;int s=0,f=1;char c=getchar();//x代表整数部分,t代表小数部分
    for (;!isdigit(c);c=getchar())
    {
        if (c=='-') f=-1;//读到负号就改变之
        if (c=='.') goto readt;//看到小数点,直接读小数部分
        if (c == ',') return ;
    }
    for (;isdigit(c)&&c!='.';c=getchar()) x=x*10+c-'0';//整数部分
    readt:for (;c=='.';c=getchar());//跳过小数点
    for (;isdigit(c);c=getchar()) t=t*10+c-'0',++s;//读小数部分,s代表小数有几位
    r=(x+t/pow(10,s))*f;//t除以10的s次方后变成小数部分
}

void input() {
    freopen("input.txt", "r", stdin);
    n = read();m=read();read();p = read();q = read();read();
//    cout << n << ' ' << m << ' ' << p << ' ' << q << endl;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++) {
            read(matrix[i * szB + j]);
//            printf("%.3f ", matrix[i*szB+j]);
        }
//        printf("\n");
//            scanf("%lf,", &matrix[i*szB+j]);
//        scanf("%lf\n", &matrix[i*szB+m-1]);
    }
    for (int i = 0; i < p; i++){
        for (int j = 0; j < q; j++)
            read(kernel[i*szL+j]);
//            scanf("%lf,", &kernel[i*szL+j]);
//        scanf("%lf\n", &kernel[i*szL+q-1]);
    }
//    printf("input done\n");
}


void output() {
    FILE* fp = fopen("output.txt", "w");
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m-1; j++)
            fprintf(fp, "%.3f,", result[(i+(p-1)/2)*szB+j+(q-1)/2]);
        fprintf(fp, "%.3f\n", result[(i+(p-1)/2)*szB+m-1+(q-1)/2]);
    }
    fclose(fp);
}

int getThreadNum()
{
    cudaDeviceProp prop;
    int count;
    cudaGetDeviceCount(&count);
    cudaGetDeviceProperties(&prop, 0);
    return prop.maxThreadsPerBlock;
}

__global__ void conv(double *matrix, double *kernel, double *result, int n, int m, int p, int q){
    int ti = threadIdx.x;
    int bi = blockIdx.x;
    int id = (bi * blockDim.x + ti);
    if(id < (n+p-1) * (m+q-1)){
        int i = id / (m+q-1);
        int j = id % (m+q-1);
        double tmp = 0.0;
        for(int k = max(0, i-p+1); k <= i; k++)
            for(int l = max(0, j-q+1); l <= j; l++)
                tmp += matrix[k*szB+l] * kernel[(i-k)*szL + j-l];
        result[i*szB+j] = tmp;
    }
}


int main()
{
    double *matrixGpu;
    double *kernelGpu;
    double *resultGpu;

    matrix = (double*)malloc(sizeof(double)*szB*szB);
    result = (double*)malloc(sizeof(double)*szB*szB);
    kernel = (double*)malloc(sizeof(double)*szL*szL);

    cudaMalloc((void**)&matrixGpu, szB*szB*sizeof(double));
    cudaMalloc((void**)&kernelGpu, szL*szL*sizeof(double));
    cudaMalloc((void**)&resultGpu, szB*szB*sizeof(double));

    input();
    cudaMemcpy(matrixGpu, matrix, 3550 *3550 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(kernelGpu, kernel, 150*150*sizeof(double), cudaMemcpyHostToDevice);

    int threadNum = getThreadNum();
    int blockNum = ((n+p-1) * (m+q-1) - 0.5) / threadNum + 1;
    
    conv<<<blockNum, threadNum>>>(matrixGpu, kernelGpu, resultGpu, n, m, p, q);
    cudaMemcpy(result, resultGpu, 3550*3550 * sizeof(double), cudaMemcpyDeviceToHost);

    output();
    cudaFree(matrixGpu);
    cudaFree(kernelGpu);
    cudaFree(resultGpu);
    
    return 0;
}
