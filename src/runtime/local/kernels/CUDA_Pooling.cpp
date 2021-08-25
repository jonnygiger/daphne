/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "CUDA_Pooling.h"

namespace Pooling {

	template<template<typename> class OP, typename DTRes, typename DTArg>
	void Forward_CUDA<OP, DTRes, DTArg>::apply(DTRes *&res, size_t& res_h, size_t& res_w,
			const DTArg *data, const size_t batch_size, const size_t num_channels, const size_t img_h, const size_t img_w,
			const size_t pool_h, const size_t pool_w, const size_t stride_h, const size_t stride_w, const size_t pad_h,
			const size_t pad_w, DCTX(dctx))
	{
		using VT = typename DTRes::VT;
		auto ctx = dctx->getCUDAContext(0);
		VT blend_alpha = 1;
		VT blend_beta = 0;
		VT* d_input;
		VT* d_res;
		size_t sizeOfDataType = sizeof(VT);
		size_t data_buf_size = batch_size * num_channels * img_h * img_w * sizeOfDataType;

		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_input), data_buf_size));
		CHECK_CUDART(cudaMemcpy(d_input, data->getValues(),  data_buf_size, cudaMemcpyHostToDevice));

		CHECK_CUDNN(cudnnSetPooling2dDescriptor(ctx->pooling_desc, OP<VT>::isMAX() ? CUDNN_POOLING_MAX :
				CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_PROPAGATE_NAN, pool_h, pool_w, pad_h, pad_w, stride_h,
				stride_w));

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->src_tensor_desc, ctx->tensor_format, ctx->data_type, batch_size,
				num_channels, img_h, img_w));

		const int tensorDims = 4;
		int tensorOuputDimA[tensorDims];
		CHECK_CUDNN(cudnnGetPoolingNdForwardOutputDim(ctx->pooling_desc, ctx->src_tensor_desc, tensorDims,
			tensorOuputDimA));

		int n = tensorOuputDimA[0]; int c = tensorOuputDimA[1];
		int h = tensorOuputDimA[2]; int w = tensorOuputDimA[3];
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ctx->dst_tensor_desc, ctx->tensor_format, ctx->data_type, n, c, h, w));

		if (res == nullptr) {
			res = DataObjectFactory::create<DTRes>(batch_size, c * h * w, false);
		}
		CHECK_CUDART(cudaMalloc(reinterpret_cast<void**>(&d_res), n * c * h * w * sizeOfDataType));

		CHECK_CUDNN(cudnnPoolingForward(ctx->getCUDNNHandle(), ctx->pooling_desc, &blend_alpha, ctx->src_tensor_desc,
										d_input, &blend_beta, ctx->dst_tensor_desc, d_res));

		CHECK_CUDART(cudaMemcpy(res->getValues(), d_res, n * c * h * w * sizeOfDataType, cudaMemcpyDeviceToHost));
		res_h = h;
		res_w = w;
		cudaFree(d_input);
		cudaFree(d_res);
	}

	template struct Forward_CUDA<AVG, DenseMatrix<float>, DenseMatrix<float>>;
	template struct Forward_CUDA<AVG, DenseMatrix<double>, DenseMatrix<double>>;

	template struct Forward_CUDA<MAX, DenseMatrix<float>, DenseMatrix<float>>;
	template struct Forward_CUDA<MAX, DenseMatrix<double>, DenseMatrix<double>>;
}

