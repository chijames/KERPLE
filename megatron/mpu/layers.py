# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import VocabUtility


def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(
    neox_args,
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    # Initialize master weight
    master_weight = torch.empty(
        output_size, input_size, dtype=torch.float, requires_grad=False
    )
    init_method(master_weight)
    master_weight = master_weight.to(dtype=neox_args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(
        master_weight, per_partition_per_stride_size, dim=partition_dim
    )
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self, neox_args, num_embeddings, embedding_dim, init_method=init.xavier_normal_
    ):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = get_model_parallel_world_size()
        # Divide the weight matrix along the vocabulary dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, get_model_parallel_rank(), self.model_parallel_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        # Allocate weights and initialize.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                init_method,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=0, stride=1
            )

    def forward(self, input_):
        if self.model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class ParallelKerpleLog(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
        self,
        neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        
        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size
        
        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

        self.cached_matrix = None
        self.cached_seq_len = None
    
    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd
    
    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p*torch.log(1+self.bias_a*diff) # log kernel
        
        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])

        return x + bias


class ParallelKerplePower(torch.nn.Module):
    """Kernelized Alibi Relative Position Bias parallelized in the heads dimension"""

    def __init__(
        self,
        neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        
        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size
        
        # Allocate weights and initialize.
        # bias_kernel = -bias_a*|m-n|^bias_p
        # weight_kernel = exp(-wei_a*|m-n|^wei_p)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        self.bias_a, self.bias_p, self.wei_a, self.wei_p = None, None, None, None
        
        if self.pos_emb.endswith('original'):
            slopes = torch.Tensor(self._get_slopes(self.heads))[
                self.model_parallel_rank * self.num_heads_per_partition : (self.model_parallel_rank + 1) * self.num_heads_per_partition
            ][:,None,None]
            slopes = slopes.to(torch.cuda.current_device()).to(neox_args.params_dtype)
            self.bias_a = Parameter(slopes, requires_grad=False)
        else:
            bias_arg, wei_arg = self.pos_emb.split('_')[-2:]
            self.bias_p = get_parameter(2, 'uniform') if 'p' in bias_arg else None
            self.bias_a = get_parameter(1, 'uniform') if 'a' in bias_arg else None
            self.wei_p = get_parameter(2, 'uniform') if 'p' in wei_arg else None
            self.wei_a = get_parameter(1, 'uniform') if 'a' in wei_arg else None

        self.cached_matrix = None
        self.cached_seq_len = None
    
    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            dd.update(get_stats('bias_a', self.bias_a))
        if self.bias_p is not None:
            self.bias_p.data = self.bias_p.data.clamp(min=self.eps, max=2)
            dd.update(get_stats('bias_p', self.bias_p))
        if self.wei_a is not None:
            self.wei_a.data = self.wei_a.data.clamp(min=self.eps)
            dd.update(get_stats('wei_a', self.wei_a))
        if self.wei_p is not None:
            self.wei_p.data = self.wei_p.data.clamp(min=self.eps, max=2)
            dd.update(get_stats('wei_p', self.wei_p))
        return dd
    
    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        
        # get bias matrix
        if self.bias_p is None and self.bias_a is None:
            bias = 0.0
        else:
            if self.bias_p is not None:
                self.bias_p.data = self.bias_p.data.clamp(min=self.eps, max=2)
                bias = diff.pow(self.bias_p)
            else:
                bias = diff
            if self.bias_a is not None:
                self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
                bias = -bias*self.bias_a
            else:
                bias = -bias

        # get weight matrix
        if self.wei_p is None and self.wei_a is None:
            wei = 1.0
        else:
            if self.wei_p is not None:
                self.wei_p.data = self.wei_p.data.clamp(min=self.eps, max=2)
                wei = diff.pow(self.wei_p)
            else:
                wei = diff
            if self.wei_a is not None:
                self.wei_a.data = self.wei_a.data.clamp(min=self.eps)
                wei = (-wei*self.wei_a).exp()
            else:
                wei = (-wei).exp()
        
        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])
            if type(wei) != float:
                wei = wei[:, seq_len_k - 1, :].view(wei.shape[0], 1, wei.shape[2])

        return x*wei + bias


class ParallelRelativePositionBias(torch.nn.Module):
    """T5 Relative Position Bias parallelized in the heads dimension

    Based on https://github.com/lucidrains/x-transformers/blob/6b93c21be0d0a679da6f7b9621d9bb638ab18428/x_transformers/x_transformers.py#L106 (14.12.2021)
    and adapted for megatron's model parallelism

    Arguments:
        scale: scaling factor for the bias
        causal: flag for causal/non-causal language modelling.
        num_buckets: number of rp buckets.
        max_distance: max distance in sequence dim for each bucket.
        heads: number of attention heads (total)
    """

    def __init__(
        self,
        neox_args,
        scale,
        causal=True,
        num_buckets=32,
        max_distance=128,
        heads=8,
        init_method=init.xavier_normal_,
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.heads = heads

        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()

        # Divide the weight matrix along the heads dimension.
        self.head_start_index, self.head_end_index = self.get_heads_range(
            self.heads, self.model_parallel_rank, self.model_parallel_size
        )
        self.num_heads_per_partition = self.head_end_index - self.head_start_index

        # Allocate weights and initialize.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_buckets,
                    self.num_heads_per_partition,
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.num_buckets,
                self.heads,
                self.num_heads_per_partition,
                partition_dim=1,
                init_method=init_method,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_buckets,
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=1, stride=1
            )
        self._q_len_cached = None
        self._k_len_cached = None
        self._rel_pos_bucket_cached = None

    @staticmethod
    def get_heads_range(global_n_heads, rank, world_size):
        per_partition_n_heads = divide(global_n_heads, world_size)
        index_f = rank * per_partition_n_heads
        index_l = index_f + per_partition_n_heads
        return index_f, index_l

    def _relative_position_bucket(
        self, relative_position, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not self.causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        self._rel_pos_bucket_cached = ret
        return self._rel_pos_bucket_cached

    def forward(self, q_len, k_len):
        if self._q_len_cached != q_len or self._k_len_cached != k_len:
            # cache bucket if first step seq len stays constant
            self._q_len_cached, self._k_len_cached = q_len, k_len
            q_pos = torch.arange(
                q_len, dtype=torch.long, device=torch.cuda.current_device()
            )
            k_pos = torch.arange(
                k_len, dtype=torch.long, device=torch.cuda.current_device()
            )
            rel_pos = k_pos[None, :] - q_pos[:, None]
            rp_bucket = self._relative_position_bucket(
                rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
            )
        else:
            rp_bucket = self._rel_pos_bucket_cached
        values = F.embedding(
            rp_bucket,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        bias = values.movedim(2, 0).unsqueeze(0)
        return bias * self.scale


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    dtype=neox_args.params_dtype,
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=0, stride=stride
            )

        if bias:
            if neox_args.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition, dtype=neox_args.params_dtype
                    )
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
                )
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def set_parallel_output(self, value: bool):
        assert isinstance(value, bool)
        self.gather_output = (
            not value
        )  # if gather_output is True, parallel output is False, so we set the opposite

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        parallel_output=False,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.parallel_output = parallel_output

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    dtype=neox_args.params_dtype,
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=1, stride=stride
            )
        if bias:
            if neox_args.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size, dtype=neox_args.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
                )
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def set_parallel_output(self, parallel_output: bool):
        assert isinstance(parallel_output, bool)
        self.parallel_output = parallel_output

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        if not self.parallel_output:
            output_ = reduce_from_model_parallel_region(output_parallel)
        else:
            output_ = output_parallel
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
