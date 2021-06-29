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
\file gk_defs.h
\brief This file contains various constants definitions

\date   Started 3/27/2007
\author George
\version\verbatim $Id: gk_defs.h 12732 2012-09-24 20:54:50Z karypis $ \endverbatim
*/

#ifndef _GK_DEFS_H_
#define _GK_DEFS_H_


#define LTERM                   (void **) 0     /* List terminator for GKfree() */

/* mopt_t types */
#define GK_MOPT_MARK            1
#define GK_MOPT_CORE            2
#define GK_MOPT_HEAP            3

#define HTABLE_EMPTY            -1
#define HTABLE_DELETED          -2
#define HTABLE_FIRST             1
#define HTABLE_NEXT              2

/* pdb corruption bit switches */
#define CRP_ALTLOCS    1
#define CRP_MISSINGCA  2
#define CRP_MISSINGBB  4
#define CRP_MULTICHAIN 8
#define CRP_MULTICA    16
#define CRP_MULTIBB    32

#define MAXLINELEN 300000

/* GKlib signals to standard signal mapping */
#define SIGMEM  SIGABRT
#define SIGERR  SIGTERM


/* CSR-related defines */
#define GK_CSR_ROW      1
#define GK_CSR_COL      2

#define GK_CSR_MAXTF    1
#define GK_CSR_SQRT     2
#define GK_CSR_POW25    3
#define GK_CSR_POW65    4
#define GK_CSR_POW75    5
#define GK_CSR_POW85    6
#define GK_CSR_LOG      7
#define GK_CSR_IDF      8
#define GK_CSR_IDF2     9
#define GK_CSR_MAXTF2   10

#define GK_CSR_COS      1
#define GK_CSR_JAC      2
#define GK_CSR_MIN      3
#define GK_CSR_AMIN     4

#define GK_CSR_FMT_CLUTO        1
#define GK_CSR_FMT_CSR          2
#define GK_CSR_FMT_METIS        3
#define GK_CSR_FMT_BINROW       4
#define GK_CSR_FMT_BINCOL       5

#define GK_GRAPH_FMT_METIS      1

#endif
