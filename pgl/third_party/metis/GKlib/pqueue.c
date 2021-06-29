// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*!
\file  pqueue.c
\brief This file implements various max-priority queues.

The priority queues are generated using the GK_MKPQUEUE macro.

\date   Started 3/27/2007
\author George
\version\verbatim $Id: pqueue.c 10711 2011-08-31 22:23:04Z karypis $ \endverbatim
*/

#include <GKlib.h>


/*************************************************************************/
/*! Create the various max priority queues */
/*************************************************************************/
#define key_gt(a, b) ((a) > (b))
GK_MKPQUEUE(gk_ipq,   gk_ipq_t,   gk_ikv_t,   int,      gk_idx_t, gk_ikvmalloc,   INT_MAX,    key_gt)
GK_MKPQUEUE(gk_i32pq, gk_i32pq_t, gk_i32kv_t, int32_t,  gk_idx_t, gk_i32kvmalloc, INT32_MAX,  key_gt)
GK_MKPQUEUE(gk_i64pq, gk_i64pq_t, gk_i64kv_t, int64_t,  gk_idx_t, gk_i64kvmalloc, INT64_MAX,  key_gt)
GK_MKPQUEUE(gk_fpq,   gk_fpq_t,   gk_fkv_t,   float,    gk_idx_t, gk_fkvmalloc,   FLT_MAX,    key_gt)
GK_MKPQUEUE(gk_dpq,   gk_dpq_t,   gk_dkv_t,   double,   gk_idx_t, gk_dkvmalloc,   DBL_MAX,    key_gt)
GK_MKPQUEUE(gk_idxpq, gk_idxpq_t, gk_idxkv_t, gk_idx_t, gk_idx_t, gk_idxkvmalloc, GK_IDX_MAX, key_gt)
#undef key_gt
