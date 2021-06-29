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

/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * omp.c
 *
 * This file contains "fake" implementations of OpenMP's runtime libraries
 *
 */

#include <GKlib.h>

#ifdef GK_NOOPENMP  /* remove those for now */
#if !defined(_OPENMP)
void omp_set_num_threads(int num_threads) { return; }
int omp_get_num_threads(void) { return 1; }
int omp_get_max_threads(void) { return 1; }
int omp_get_thread_num(void) { return 0; }
int omp_get_num_procs(void) { return 1; }
int omp_in_parallel(void) { return 0; }
void omp_set_dynamic(int num_threads) { return; }
int omp_get_dynamic(void) { return 0; }
void omp_set_nested(int nested) { return; }
int omp_get_nested(void) { return 0; }
#endif
#endif
