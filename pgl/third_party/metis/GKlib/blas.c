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
\file blas.c
\brief This file contains GKlib's implementation of BLAS-like routines

The BLAS routines that are currently implemented are mostly level-one.
They follow a naming convention of the type gk_[type][name], where
[type] is one of c, i, f, and d, based on C's four standard scalar
datatypes of characters, integers, floats, and doubles.

These routines are implemented using a generic macro template,
which is used for code generation.

\date   Started 9/28/95
\author George
\version\verbatim $Id: blas.c 11848 2012-04-20 13:47:37Z karypis $ \endverbatim
*/

#include <GKlib.h>



/*************************************************************************/
/*! Use the templates to generate BLAS routines for the scalar data types */
/*************************************************************************/
GK_MKBLAS(gk_c,   char,     int)
GK_MKBLAS(gk_i,   int,      int)
GK_MKBLAS(gk_i32, int32_t,  int32_t)
GK_MKBLAS(gk_i64, int64_t,  int64_t)
GK_MKBLAS(gk_z,   ssize_t,  ssize_t)
GK_MKBLAS(gk_f,   float,    float)
GK_MKBLAS(gk_d,   double,   double)
GK_MKBLAS(gk_idx, gk_idx_t, gk_idx_t)
