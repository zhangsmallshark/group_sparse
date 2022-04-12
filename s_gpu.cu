#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <malloc.h>
#include <time.h>
#include <sys/types.h>
#include <errno.h>
#include<vector>
#include<algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cudnn.h>

// printf("%d, %d, %d, %d, %d\n", o_idx0, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
T* readArray(const std::string& filename, int& arr_len)
{
    std::ifstream in_s(filename.c_str(), std::ios::binary | std::ios::in);

    if (in_s.fail()) {
        std::cerr << "fail to open " << filename << std::endl;
        exit(1);
    }

    in_s.seekg(0, std::ios::end);
    arr_len = in_s.tellg() / 4;
    std::cout << "array length " << arr_len << std::endl;

    in_s.seekg(0, std::ios::beg);
    T* arr = new T[arr_len];
    // in_s.read((char*) &arr, sizeof arr); Error
    in_s.read(reinterpret_cast<char*>(arr), std::streamsize(arr_len * sizeof(T)));
    in_s.close();
    return arr;
}

template <typename T>
void writeArray(const std::string& filename, T* const _a, size_t const arr_len)
{
    std::ofstream out_s(filename.c_str(), std::ios::binary | std::ios::out);
    // if (not out_s.is_open()) return;
    out_s.write(reinterpret_cast<const char*>(_a), std::streamsize(arr_len * sizeof(T)));
    out_s.close();
}

template <typename T>
void checkArray(T* a, T* b, size_t d_len) {
	for (int i = 0; i < d_len; i++) {
		if ((std::abs(a[i] - b[i]) > 0.04) || (i < 16)) {
		// if ((a[i] != b[i]) && (i < 32) ) {

			std::cerr << "Error " << i << " " << a[i] << " " << b[i] << std::endl;
		}
	}
}

void inner(const float* a, const float* b, float* c, int m, int n, int k) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float c_reg = 0.0;
			for (int x = 0; x < k; x++) {
				c_reg += a[i*n+x] * b[x*k+j];
			}
			c[i*k+j] = c_reg;
		}
	}
}
__global__ void columnWise(const float* __restrict__ filter, const float* __restrict__ input, float* output) {
	const int kRowsWeight = 32;
	const int kColsWeight = 32;
	const int kColsInput = 32;

	int t_x = threadIdx.x + blockIdx.x * blockDim.x;

	float res[kColsWeight] = {0};
	for (int i = 0; i < kColsWeight; i++) {
		float i_reg0 = input[i*kColsInput+t_x]; 
		for (int j = 0; j < kRowsWeight; j++) {
			res[j] += filter[j*kColsWeight+i] * i_reg0;
		}
	}

	for (int i = 0; i < kColsWeight; i++) {
		output[i*kColsInput+t_x] = res[i];
	}
}

// int main()
// {
// 	float filter[32*32] = {0};
// 	float input[32*32] = {0};
// 	float output[32*32] = {0};

// 	float* d_filter;
// 	float* d_input;
// 	float* d_output;

// 	for (int i = 0; i < 1024; i++) {
// 		filter[i] = i;
// 		input[i] = i;
// 	}

// 	cudaMalloc(&d_filter, 1024*sizeof(float));
// 	cudaMemcpy(d_filter, filter, 1024*sizeof(float), cudaMemcpyHostToDevice);

// 	cudaMalloc(&d_input, 1024*sizeof(float));
// 	cudaMemcpy(d_input, input, 1024*sizeof(float), cudaMemcpyHostToDevice);

// 	cudaMalloc(&d_output, 1024*sizeof(float));

//     dim3 blocks_per_grid(1);
// 	dim3 threads_per_block(32);
// 	columnWise<<<blocks_per_grid, threads_per_block>>>(d_filter, d_input, d_output);

// 	cudaDeviceSynchronize();
// 	cudaMemcpyAsync(output, d_output, 1024*sizeof(float), cudaMemcpyDeviceToHost);

// 	float refer_output[32*32] = {0};
// 	inner(filter, input, refer_output, 32, 32, 32);

// 	checkArray<float>(refer_output, output, 1024);

// 	// for (int o_i = 0; o_i < 16; o_i++) {
// 	// 	std::cout << o_i << " " << output[o_i] << std::endl;
// 	// }
// }


inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) << std::endl;
        exit(-1);
    }
}

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

class CUDNN_CONVOLUTION_FWD_ALGO_GEMM_CONV {
public:
    unsigned int N;
    unsigned int inC;
    unsigned int H;
    unsigned int W;
    unsigned int outC;
    unsigned int kH;
    unsigned int kW;
    unsigned int outH;
    unsigned int outW;
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnHandle_t convCudnn;
    void* d_workspace{nullptr};
    size_t workspace_bytes{0};
    cudnnTensorDescriptor_t convInputDescriptor;
    cudnnTensorDescriptor_t convOutputDescriptor;
    cudnnFilterDescriptor_t convKernelDescriptor;
    cudnnConvolutionDescriptor_t convDesc;
    float *output;
    float *filter;

	void initialize(unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w,
		unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad);

    float* forward(float* filter, float* input);
};

void CUDNN_CONVOLUTION_FWD_ALGO_GEMM_CONV::initialize(unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w, 
	unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad) {
    this->N = batch_size;
    this->inC = in_channels;
    this->H = feature_h;
    this->W = feature_w;
    this->outC = out_channels;
    this->kH = kernel_h;
    this->kW = kernel_w;
    this->outH = (H + 2 * pad - kH) / stride + 1;
    this->outW = (W + 2 * pad - kW) / stride + 1;

    cudaMalloc(&this->output, sizeof(float)*N*outC*outH*outW);
    cudnnCreate(&convCudnn);
    cudnnCreateTensorDescriptor(&convInputDescriptor);
    cudnnSetTensor4dDescriptor(convInputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*data_type=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/N,
            /*channels=*/inC,
            /*image_height=*/H,
            /*image_width=*/W);
    cudnnCreateFilterDescriptor(&convKernelDescriptor);
    cudnnSetFilter4dDescriptor(convKernelDescriptor,
            /*data_type=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/outC,
            /*in_channels=*/inC,
            /*kernel_height=*/kH,
            /*kernel_width=*/kW);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc,
            /*pad_height=*/pad,
            /*pad_width=*/pad,
            /*vertical_stride=*/stride,
            /*horizontal_stride=*/stride,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    int out_batch_size{0}, channels{0}, height{0}, width{0};
    cudnnGetConvolution2dForwardOutputDim(convDesc,
                                          convInputDescriptor,
                                          convKernelDescriptor,
                                          &out_batch_size,
                                          &channels,
                                          &height,
                                          &width);
    cudnnCreateTensorDescriptor(&convOutputDescriptor);
    cudnnSetTensor4dDescriptor(convOutputDescriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*data_type=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/N,
            /*channels=*/outC,
            /*image_height=*/outH,
            /*image_width=*/outW);
    cudnnGetConvolutionForwardWorkspaceSize(convCudnn,
                                            convInputDescriptor,
                                            convKernelDescriptor,
                                            convDesc,
                                            convOutputDescriptor,
                                            CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                                            &workspace_bytes);

    cudaMalloc(&d_workspace, workspace_bytes);
}

float * CUDNN_CONVOLUTION_FWD_ALGO_GEMM_CONV::forward(float* filter, float *input) {
    cudaMemset(output, 0, N*outC*outH*outW*sizeof(float));
    checkCUDNN(cudnnConvolutionForward(convCudnn,
                                       &alpha,
                                       convInputDescriptor,
                                       input,
                                       convKernelDescriptor,
                                       filter,
                                       convDesc,
                                       CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       convOutputDescriptor,
                                       output));
    return output;
}

// Inner product
void implicitConvCpu(const float* __restrict__ filter, const float* __restrict__ input, float* output, unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w, 
	unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad) {
	int out_h = (feature_h + 2 * pad - kernel_h) / stride + 1;
	int out_w = (feature_w + 2 * pad - kernel_w) / stride + 1;

	int M = out_channels;
	int N = batch_size * out_h * out_h;
	int X = in_channels * kernel_h * kernel_w;

	for (int i = 0; i < M; i++) {
	    int o_c = i;
		for (int j = 0; j < N; j++) {
	        float res = 0.0;
	        int n = j / (out_h * out_w);
	        int j_res = j % (out_h * out_w);
	        int o_h = j_res / out_w;
	        int o_w = j_res % out_w;
	        for (int x = 0; x < X; x++) {
	            int i_c = x / (kernel_h * kernel_w);
	            int x_res = x % (kernel_h * kernel_w);
	            int k_h = x_res / kernel_w;
	            int k_w = x_res % kernel_w;
	            int i_h = o_h * stride - pad + k_h;
	            int i_w = o_w * stride - pad + k_w;
	            int f_idx = o_c * in_channels * kernel_h * kernel_w + i_c * kernel_h * kernel_w + k_h * kernel_w + k_w;
	            if ((i_h >= 0) && (i_w >= 0) && (i_h < feature_h) && (i_w < feature_w)) {
		            int i_idx = n * in_channels * feature_h * feature_w + i_c * feature_h * feature_w + i_h * feature_w + i_w;
		            res = res + filter[f_idx] * input[i_idx];
	            }
	        }
	       	int o_idx = n * out_channels * out_h * out_w + o_c * out_h * out_w + o_h * out_w + o_w; 
	       	output[o_idx] = res;
		}
	}
}

// Row-wise product
void implicitSparseConvCpu0(const float* __restrict__ filter, const float* __restrict__ input, float* output, unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w, 
	unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad) {
	int out_h = (feature_h + 2 * pad - kernel_h) / stride + 1;
	int out_w = (feature_w + 2 * pad - kernel_w) / stride + 1;

	int M = out_channels;
	int N = batch_size * out_h * out_h;
	int X = in_channels * kernel_h * kernel_w;

	for (int i = 0; i < M; i++) {
	    int o_c = i;
	    float* res = new float[N]();
        for (int x = 0; x < X; x++) {
            int i_c = x / (kernel_h * kernel_w);
            int x_res = x % (kernel_h * kernel_w);
            int k_h = x_res / kernel_w;
            int k_w = x_res % kernel_w;
            int f_idx = o_c * in_channels * kernel_h * kernel_w + i_c * kernel_h * kernel_w + k_h * kernel_w + k_w;
            float filter_val = filter[f_idx];
			for (int j = 0; j < N; j++) {
		        int n = j / (out_h * out_w);
		        int j_res = j % (out_h * out_w);
		        int o_h = j_res / out_w;
		        int o_w = j_res % out_w;
	            int i_h = o_h * stride - pad + k_h;
	            int i_w = o_w * stride - pad + k_w;
	            if ((i_h >= 0) && (i_w >= 0) && (i_h < feature_h) && (i_w < feature_w)) {
		            int i_idx = n * in_channels * feature_h * feature_w + i_c * feature_h * feature_w + i_h * feature_w + i_w;
		            res[j] = res[j] + filter_val * input[i_idx];
	            }
			}
        }

		for (int j = 0; j < N; j++) {
	        int n = j / (out_h * out_w);
	        int j_res = j % (out_h * out_w);
	        int o_h = j_res / out_w;
	        int o_w = j_res % out_w;
			int o_idx = n * out_channels * out_h * out_w + o_c * out_h * out_w + o_h * out_w + o_w; 
			output[o_idx] = res[j];
		}
	}	
}

// Row-wise product
void implicitSparseConvCpu1(const int* __restrict__ filter_ptr, const int* __restrict__ filter_indices, const float* __restrict__ filter_data, const float* __restrict__ input, float* output, unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w, 
	unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad) {
	int out_h = (feature_h + 2 * pad - kernel_h) / stride + 1;
	int out_w = (feature_w + 2 * pad - kernel_w) / stride + 1;

	int M = out_channels;
	int N = batch_size * out_h * out_h;
	int X = in_channels * kernel_h * kernel_w;

	for (int i = 0; i < M; i++) {
	    int o_c = i;
	    float* res = new float[N]();

	    int offset = filter_ptr[i];
	    int Y = filter_ptr[i+1] - filter_ptr[i];
	    for(int y = 0; y < Y; y++) {
	    	int x = filter_indices[offset+y];
            int i_c = x / (kernel_h * kernel_w);
            int x_res = x % (kernel_h * kernel_w);
            int k_h = x_res / kernel_w;
            int k_w = x_res % kernel_w;
            float filter_val = filter_data[offset+y];

			for (int j = 0; j < N; j++) {
		        int n = j / (out_h * out_w);
		        int j_res = j % (out_h * out_w);
		        int o_h = j_res / out_w;
		        int o_w = j_res % out_w;
	            int i_h = o_h * stride - pad + k_h;
	            int i_w = o_w * stride - pad + k_w;
	            if ((i_h >= 0) && (i_w >= 0) && (i_h < feature_h) && (i_w < feature_w)) {
		            int i_idx = n * in_channels * feature_h * feature_w + i_c * feature_h * feature_w + i_h * feature_w + i_w;
		            res[j] = res[j] + filter_val * input[i_idx];
	            }
			}
        }

		for (int j = 0; j < N; j++) {
	        int n = j / (out_h * out_w);
	        int j_res = j % (out_h * out_w);
	        int o_h = j_res / out_w;
	        int o_w = j_res % out_w;
			int o_idx = n * out_channels * out_h * out_w + o_c * out_h * out_w + o_h * out_w + o_w; 
			output[o_idx] = res[j];
		}
	}	
}

// const int kTileM = 1;
// const int kTileN = 32;
// const int kTileX = 32;
// constexpr int kTileFilter = kTileM * kTileX;

#define kTileM 1
#define kTileN 32
#define kTileX 32

// Inner product
__global__ void implicitSparseConvCuda0(const int* __restrict__ filter_ptr, const int* __restrict__ filter_indices, const float* __restrict__ filter_data, const float* __restrict__ input, float* output, unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w, 
	unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad) {
	int out_h = (feature_h + 2 * pad - kernel_h) / stride + 1;
	int out_w = (feature_w + 2 * pad - kernel_w) / stride + 1;

    __shared__ float filter_data_tile[kTileX];
    __shared__ int filter_indices_tile[kTileX];
    float input_tile[kTileX] = {0}; 
    float res = 0;

	int m_idx = blockIdx.y;
	int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int t_idx = threadIdx.x;

    int o_c = m_idx;
	int n = n_idx / (out_h * out_w);
	int n_res = n_idx % (out_h * out_w);
	int o_h = n_res / out_w;
	int o_w = n_res % out_w;

    int offset = filter_ptr[m_idx];
    int cur_nnzs = filter_ptr[m_idx+1] - offset;

    for(int i = 0; i < cur_nnzs; i += kTileX) {
        filter_data_tile[t_idx] = filter_data[offset+i+t_idx];
        filter_indices_tile[t_idx] = filter_indices[offset+i+t_idx];
    	__syncthreads();

        #pragma unroll
	    for (int j = 0; j < kTileX; j++) {
		   	int x = filter_indices_tile[j];
		    int i_c = x / (kernel_h * kernel_w);
		    int x_res = x % (kernel_h * kernel_w);
		    int k_h = x_res / kernel_w;
		    int k_w = x_res % kernel_w;
		    int i_h = o_h * stride - pad + k_h;
		    int i_w = o_w * stride - pad + k_w;

		    if ((i_h >= 0) && (i_w >= 0) && (i_h < feature_h) && (i_w < feature_w)) {
		        int i_idx = n * in_channels * feature_h * feature_w + i_c * feature_h * feature_w + i_h * feature_w + i_w;
                res = res + filter_data_tile[j] * input[i_idx];
		        // input_tile[j] = input[i_idx];
		    } 
      //       else {
		    // 	input_tile[j] = 0.0;
		    // }
	    }
    	// __syncthreads();

    	// for(int x = 0; x < kTileX; x++ ) {
    	// 	res = res + filter_data_tile[x] * input_tile[x];
    	// }
    	// __syncthreads();
   }

   	// __syncthreads();

   	int o_idx = n * out_channels * out_h * out_w + o_c * out_h * out_w + o_h * out_w + o_w; 
   	output[o_idx] = res;
}

// Column-wise product
__global__ void implicitSparseConvCuda1(const int ptr_start, const int* __restrict__ all_filter_ptr, const int* __restrict__ all_filter_map, const int* __restrict__ filter_indices, const float* __restrict__ filter_data, const float* __restrict__ input, float* output, unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w, 
    unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad) {
    int out_h = (feature_h + 2 * pad - kernel_h) / stride + 1;
    int out_w = (feature_w + 2 * pad - kernel_w) / stride + 1;

    const int* filter_ptr = all_filter_ptr + ptr_start;
    const int* filter_map = all_filter_map + ptr_start;

    // __shared__ float filter_data_tile[kTileX];
    // __shared__ int filter_indices_tile[kTileX];
    float res[32] = {0.0};

    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t_idx = threadIdx.x;

    int n = n_idx / (out_h * out_w);
    int n_res = n_idx % (out_h * out_w);
    int o_h = n_res / out_w;
    int o_w = n_res % out_w;

    int offset = filter_ptr[0];
    int cur_nnzs = filter_ptr[1] - offset;

    float input_reg = 0.0;
    for(int i = 0; i < cur_nnzs; i++) {
        int x = filter_indices[offset+i];
        int i_c = x / (kernel_h * kernel_w);
        int x_res = x % (kernel_h * kernel_w);
        int k_h = x_res / kernel_w;
        int k_w = x_res % kernel_w;
        int i_h = o_h * stride - pad + k_h;
        int i_w = o_w * stride - pad + k_w;

        if ((i_h >= 0) && (i_w >= 0) && (i_h < feature_h) && (i_w < feature_w)) {
            int i_idx = n * in_channels * feature_h * feature_w + i_c * feature_h * feature_w + i_h * feature_w + i_w;
            input_reg = input[i_idx];
        } else {
            input_reg = 0.0;
        } 

        for(int j = 0; j < 32; j++) {
            int m_idx = j;
            res[j] = res[j] + filter_data[m_idx*cur_nnzs+i] * input_reg;
        }
    }

    for (int k = 0; k < 32; k++) {
        int o_c = filter_map[k];
        int o_idx = n * out_channels * out_h * out_w + o_c * out_h * out_w + o_h * out_w + o_w; 
        output[o_idx] = res[k];
    }
}

// Column-wise product
__global__ void implicitSparseConvCuda2(const int ptr_start, const int* __restrict__ all_filter_ptr, const int* __restrict__ all_filter_map, const int* __restrict__ filter_indices, const float* __restrict__ filter_data, const float* __restrict__ input, float* output, unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w, 
    unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad) {
    int out_h = (feature_h + 2 * pad - kernel_h) / stride + 1;
    int out_w = (feature_w + 2 * pad - kernel_w) / stride + 1;

    const int* filter_ptr = all_filter_ptr + ptr_start;
    const int* filter_map = all_filter_map + ptr_start;

    // __shared__ float filter_data_tile[kTileX];
    // __shared__ int filter_indices_tile[kTileX];

    float filter_data_tile[32] = {0.0};
    float res[32] = {0.0};

    int n_idx = blockIdx.x * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    int t_idx = threadIdx.x;

    int n = n_idx / (out_h * out_w);
    int n_res = n_idx % (out_h * out_w);
    int o_h = n_res / out_w;
    int o_w = n_res % out_w;

    int offset = filter_ptr[0];
    int cur_nnzs = filter_ptr[1] - offset;
    int pos_i = 0;
    float input_reg = 0.0;

    // float tmp0 = 0.0;
    // float tmp1 = 0.0;

    for(int i = 0; i < cur_nnzs; i++) {

        // tmp0 += 1.0;
        // tmp1 += 1.0;

        if ((i & 31) == 0) {
            pos_i = i;
            if (t_idx < cur_nnzs - i) {
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    filter_data_tile[j] = filter_data[j*cur_nnzs+i+t_idx];
                }
            } 
        }

        int x = filter_indices[offset+i];
        int i_c = x / (kernel_h * kernel_w);
        int x_res = x % (kernel_h * kernel_w);
        int k_h = x_res / kernel_w;
        int k_w = x_res % kernel_w;
        int i_h = o_h * stride - pad + k_h;
        int i_w = o_w * stride - pad + k_w;

        if ((i_h >= 0) && (i_w >= 0) && (i_h < feature_h) && (i_w < feature_w)) {
            int i_idx = n * in_channels * feature_h * feature_w + i_c * feature_h * feature_w + i_h * feature_w + i_w;
            input_reg = input[i_idx];
        } else {
            input_reg = 0.0;
        } 

        #pragma unroll
        for(int j = 0; j < 32; j++) {
            float val = __shfl_sync(0xFFFFFFFF, filter_data_tile[j], i-pos_i);
            res[j] = res[j] + val * input_reg;
        }
    }

    #pragma unroll
    for (int k = 0; k < 32; k++) {
        int o_c = filter_map[k];
        int o_idx = n * out_channels * out_h * out_w + o_c * out_h * out_w + o_h * out_w + o_w; 
        output[o_idx] = res[k];
    }
}

void imToCol(float* im, float* col, unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w, 
    unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad) {
    cudnnHandle_t im_to_col;
    cudnnTensorDescriptor_t im_descriptor;
    cudnnFilterDescriptor_t f_descriptor;
    cudnnConvolutionDescriptor_t conv_descriptor;

    cudnnCreate(&im_to_col);

    // create im descriptor
    cudnnCreateTensorDescriptor(&im_descriptor);
    cudnnSetTensor4dDescriptor(im_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*data_type=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/batch_size,
            /*channels=*/in_channels,
            /*image_height=*/feature_h,
            /*image_width=*/feature_w);

    // create filter descriptor
    cudnnCreateFilterDescriptor(&f_descriptor);
    cudnnSetFilter4dDescriptor(f_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 
        out_channels, in_channels, kernel_h, kernel_w);

    // create convolution descriptor
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor, pad, pad, stride, stride, 1, 1, 
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    cudnnIm2Col(im_to_col, im_descriptor, im, f_descriptor, conv_descriptor, col);
}


// Inner product
__global__ void SparseConvCuda0(const int* __restrict__ filter_ptr, const int* __restrict__ filter_indices, const float* __restrict__ filter_data, const float* __restrict__ input, float* output, unsigned int batch_size, unsigned int in_channels, unsigned int feature_h, unsigned int feature_w, 
    unsigned int out_channels, unsigned int kernel_h, unsigned int kernel_w, unsigned int stride, unsigned int pad) {
    int out_h = (feature_h + 2 * pad - kernel_h) / stride + 1;
    int out_w = (feature_w + 2 * pad - kernel_w) / stride + 1;

    int M = out_channels;
    int N = batch_size * out_h * out_h;
    int X = in_channels * kernel_h * kernel_w;

    __shared__ float filter_data_tile[2][kTileX];
    __shared__ float input_tile[kTileX][kTileX];

    int filter_indices_tile[kTileX] = {0};
    // float res0[32] = { 0 };
    // float res1[32] = { 0 };

    float res0 = 0;
    float res1 = 0;

    float tmp0 = 0;
    float tmp1 = 0;

    int m_idx = blockIdx.y * 2;
    int n_idx = blockIdx.x * blockDim.x;
    int t_idx = threadIdx.x;

    int offset = filter_ptr[0];
    int cur_nnzs = filter_ptr[1] - offset;
    #pragma unroll
    for(int i = 0; i < 64; i += kTileX) {
        // filter_indices_tile[t_idx] = filter_indices[offset+i+t_idx];
        // filter_data_tile[0][t_idx] = filter_data[offset+m_idx*cur_nnzs+i+t_idx];
        // filter_data_tile[1][t_idx] = filter_data[offset+(m_idx+1)*cur_nnzs+i+t_idx];

        // tmp0 = filter_data[offset+m_idx*cur_nnzs+i+t_idx];
        // tmp1 = filter_data[offset+(m_idx+1)*cur_nnzs+i+t_idx];

        tmp0 += 1;
        tmp1 += 1;

        // filter_data_tile[2][t_idx] = filter_data[offset + (m_idx + 2) * cur_nnzs + i + t_idx];
        // filter_data_tile[3][t_idx] = filter_data[offset + (m_idx + 3) * cur_nnzs + i + t_idx];

         // __syncthreads();

        // #pragma unroll
        // for (int j = 0; j < 32; j++) {
        //     // int col_f = filter_indices_tile[j];
        //     input_tile[j][t_idx] = input[(i+j)*N+n_idx+t_idx];
        // }
        // __syncthreads();

        // #pragma unroll
        // for (int j = 0; j < kTileX; j++) {
        //     // res0[t_idx] += filter_data_tile[0][j] * input_tile[j][t_idx];
        //     // res1[t_idx] += filter_data_tile[1][j] * input_tile[j][t_idx];

        //    res0 += filter_data_tile[0][j] * input_tile[j][t_idx];
        //    res1 += filter_data_tile[1][j] * input_tile[j][t_idx];
        // }
        // __syncthreads();
   }

    // __syncthreads();

    // int o_idx = m_idx * N + n_idx + t_idx; 
    // output[o_idx] = res0[t_idx];

    // o_idx = (m_idx + 1) * N + n_idx + t_idx; 
    // output[o_idx] = res1[t_idx];

    // o_idx = (m_idx + 2) * N + n_idx + t_idx;
    // output[o_idx] = res[2][t_idx];

    // o_idx = (m_idx + 3) * N + n_idx + t_idx;
    // output[o_idx] = res[3][t_idx];
}



void genUniqueRand(int max_num, int num_need, int* rand_num) {
	std::vector<int> temp;
	for (int i = 0; i < max_num; ++i) {
		temp.push_back(i);
	}
	random_shuffle(temp.begin(), temp.end());
	for (int i = 0; i < num_need; i++) {
		// rand_num[i] = temp[i];
		rand_num[temp[i]] = 1;
	}
}

void genSparseFilter(float density, unsigned int out_channels, unsigned int in_channels, unsigned int kernel_h, unsigned int kernel_w, 
	float* filter, int* filter_ptr, int* filter_indices, float* filter_data) {
	int M = out_channels;
	int X = in_channels * kernel_h * kernel_w;

    int total_num = M * X;
    int total_nnzs = static_cast<int>(total_num*density);
    int nnzs_per_row = static_cast<int>(X*density);
    int cur_nnzs = 0;
    int count = 0; 
    filter_ptr[0] = 0;
    srand((unsigned)time(NULL));

	for (int i = 0; i < M; i++) {
	    int o_c = i;
	    int* rand_num = new int[X]();
	    genUniqueRand(X, nnzs_per_row, rand_num);

        for (int x = 0; x < X; x++) {
        	// if(rand() / (double)RAND_MAX < ((double)total_nnzs - (double)cur_nnzs) / ((double)total_num - (double)count)){
        	// if (rand_num[x] == 1) {
        	if (x < nnzs_per_row) {
        		// if (cur_nnzs == total_nnzs) {
        		// 	break;
        		// }

	            int i_c = x / (kernel_h * kernel_w);
	            int x_res = x % (kernel_h * kernel_w);
	            int k_h = x_res / kernel_w;
	            int k_w = x_res % kernel_w;
	            int f_idx = o_c * in_channels * kernel_h * kernel_w + i_c * kernel_h * kernel_w + k_h * kernel_w + k_w;
	            filter[f_idx] = (x % 64) * 0.1;
	            filter_indices[cur_nnzs] = x;
	            filter_data[cur_nnzs] = (x % 64) * 0.1;
				cur_nnzs++;
        	}
        	count++;
        }
        filter_ptr[i+1] = cur_nnzs;
    }

    std::cout << "final nonzeros " << cur_nnzs << "  final density " << static_cast<float>(cur_nnzs) / total_num << std::endl;
}


int main(int argc, char *argv[]){
    // unsigned int batch_size = atoi(argv[1]);
    // unsigned int in_channels = atoi(argv[2]);
    // unsigned int feature_h = atoi(argv[3]);
    // unsigned int feature_w = atoi(argv[4]);
    // unsigned int out_channels = atoi(argv[5]);

    unsigned int batch_size = 16;
    unsigned int in_channels= 8;
    unsigned int feature_h =  64;
    unsigned int feature_w =  64;
    unsigned int out_channels = 32;

	unsigned int kernel_h = 3;
	unsigned int kernel_w = 3; 
	unsigned int stride = 1;
	unsigned int pad = 1;
	int out_h = (feature_h + 2 * pad - kernel_h) / stride + 1;
	int out_w = (feature_w + 2 * pad - kernel_w) / stride + 1;
	int M = out_channels;
	int N = batch_size * out_h * out_h;
	int X = in_channels * kernel_h * kernel_w;
	
    unsigned int filter_size = out_channels * in_channels * kernel_h * kernel_w; //filter
    float* filter = (float*)malloc(filter_size*sizeof(float));
    for(unsigned int i=0; i < filter_size; ++i){
        // filter[i] = (i % 64) * 0.1;
        filter[i] = 0.0;
    }
	float density = 0.89;
	// int total_nnzs = static_cast<int>(filter_size*density);
	int nnzs_per_row = static_cast<int>(X*density);
	int total_nnzs = nnzs_per_row * M;
	//int* filter_ptr = new int[out_channels+1];
	int filter_ptr[33] = { 0 };
	int* filter_indices = new int[total_nnzs];
	float* filter_data = new float[total_nnzs];
	genSparseFilter(density, out_channels, in_channels, kernel_h, kernel_w, filter, filter_ptr, filter_indices, filter_data);

    float* d_filter;
    cudaMalloc(&d_filter, filter_size*sizeof(float));
    cudaMemcpy(d_filter, filter, filter_size*sizeof(float), cudaMemcpyHostToDevice);

    unsigned int input_size = batch_size * in_channels * feature_h * feature_w;
    float* input = (float*)malloc(input_size*sizeof(float));
    for (unsigned int i=0; i < input_size; i++) {
        input[i] = (i % 64) * 0.1;
    }
    float* d_input;
    cudaMalloc(&d_input, input_size*sizeof(float));
    cudaMemcpy(d_input, input, input_size*sizeof(float), cudaMemcpyHostToDevice);

    float* d_col;
    cudaMalloc(&d_col, X*N*sizeof(float));
    cudaMemset(d_col, 0, X*N*sizeof(float));
    imToCol(d_input, d_col, batch_size, in_channels, feature_h, feature_w, out_channels, kernel_h, kernel_w, stride, pad);
    // float* col_cpu = new float[X*N]();
    // cudaMemcpy(col_cpu, d_col, X * N * sizeof(float), cudaMemcpyDeviceToHost);

    // int col_size;
    // float* col_py = readArray<float>("../data/col.bin", col_size);
    // cudaMemcpy(d_col, col_py, col_size*sizeof(float), cudaMemcpyHostToDevice);

	unsigned int output_size = batch_size * out_channels * out_h * out_w; 
	float* output_cudnn = (float*)malloc(output_size*sizeof(float));
    float* d_output_cudnn;

    CUDNN_CONVOLUTION_FWD_ALGO_GEMM_CONV conv_cudnn;
	conv_cudnn.initialize(batch_size, in_channels, feature_h, feature_w, out_channels, kernel_h, kernel_w, stride, pad);
	// Warm up GPU
    // d_output_cudnn = conv_cudnn.forward(d_filter, d_input);
     float inference_time;
     cudaEvent_t event_start;
     cudaEvent_t event_stop;
     cudaEventCreate(&event_start);
     cudaEventCreate(&event_stop);

    cudaEventRecord(event_start);
    d_output_cudnn = conv_cudnn.forward(d_filter, d_input);
    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&inference_time, event_start, event_stop);
    std::cout<<"CUDNN_CONVOLUTION_FWD_ALGO_GEMM_CONV, " << inference_time << std::endl;

    cudaMemcpy(output_cudnn, d_output_cudnn, output_size*sizeof(float), cudaMemcpyDeviceToHost);

    // float* output_conv_cpu = (float*)malloc(output_size*sizeof(float));
	// implicitConvCpu(filter, input, output_conv_cpu, batch_size, in_channels, feature_h, feature_w, out_channels, kernel_h, kernel_w, stride, pad);
	// implicitSparseConvCpu0(filter, input, output_conv_cpu, batch_size, in_channels, feature_h, feature_w, out_channels, kernel_h, kernel_w, stride, pad);
	// implicitSparseConvCpu1(filter_ptr, filter_indices, filter_data, input, output_conv_cpu, batch_size, in_channels, feature_h, feature_w,
	// 	out_channels, kernel_h,kernel_w, stride, pad);

    int* d_filter_ptr;
    cudaMalloc(&d_filter_ptr, (out_channels+1)*sizeof(float));
    cudaMemcpy(d_filter_ptr, filter_ptr, (out_channels+1)*sizeof(float), cudaMemcpyHostToDevice);

    int* filter_map = new int[out_channels];
    for (int i = 0; i < 32; i++) {
        filter_map[i] = i;
    }
    int* d_filter_map;
    cudaMalloc(&d_filter_map, out_channels*sizeof(float));
    cudaMemcpy(d_filter_map, filter_map, out_channels*sizeof(float), cudaMemcpyHostToDevice);

    int* d_filter_indices;
    cudaMalloc(&d_filter_indices, total_nnzs*sizeof(float));
    cudaMemcpy(d_filter_indices, filter_indices, total_nnzs*sizeof(float), cudaMemcpyHostToDevice);

    float* d_filter_data;
    cudaMalloc(&d_filter_data, total_nnzs*sizeof(float));
    cudaMemcpy(d_filter_data, filter_data, total_nnzs*sizeof(float), cudaMemcpyHostToDevice);

	// float* output_cuda = (float*)malloc(output_size*sizeof(float));
	float* output_cuda = new float[output_size]();
    float* d_output_cuda;
    cudaMalloc(&d_output_cuda, output_size*sizeof(float));

    // dim3 blocks_per_grid(N / kTileN, M / 2);
	dim3 threads_per_block(kTileN, 8);
    cudaEventRecord(event_start);
	// implicitSparseConvCuda0<<<blocks_per_grid, threads_per_block>>>(d_filter_ptr, d_filter_indices, d_filter_data, d_input, d_output_cuda, batch_size, in_channels, feature_h, feature_w,
	// 	out_channels, kernel_h,kernel_w, stride, pad);

    dim3 blocks_per_grid(N / (kTileN * 16 * 8)); 
    implicitSparseConvCuda2<<<blocks_per_grid, threads_per_block>>>(0, d_filter_ptr, d_filter_map, d_filter_indices, d_filter_data, d_input, d_output_cuda, batch_size, in_channels, feature_h, feature_w,
       out_channels, kernel_h,kernel_w, stride, pad);

    // SparseConvCuda0<<<blocks_per_grid, threads_per_block>>>(d_filter_ptr, d_filter_indices, d_filter_data, d_col, d_output_cuda, batch_size, in_channels, feature_h, feature_w,
    //     out_channels, kernel_h,kernel_w, stride, pad);

    cudaEventRecord(event_stop);
    cudaEventSynchronize(event_stop);
    cudaEventElapsedTime(&inference_time, event_start, event_stop);
    std::cout<<"SparseConvCuda, " << inference_time << std::endl;

	gpuErrChk( cudaMemcpy(output_cuda, d_output_cuda, output_size*sizeof(float), cudaMemcpyDeviceToHost) );

	// checkArray<float>(output_cudnn, output_cuda, output_size);

	std::cout<<"Finish" << std::endl;
    // free(filter);
}
