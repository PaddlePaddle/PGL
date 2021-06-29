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
\file gk_arch.h
\brief This file contains various architecture-specific declerations

\date   Started 3/27/2007
\author George
\version\verbatim $Id: gk_arch.h 10711 2011-08-31 22:23:04Z karypis $ \endverbatim
*/

#ifndef _GK_ARCH_H_
#define _GK_ARCH_H_

/*************************************************************************
* Architecture-specific differences in header files
**************************************************************************/
#ifdef LINUX
#if !defined(__USE_XOPEN)
#define __USE_XOPEN
#endif
#if !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 600
#endif
#if !defined(__USE_XOPEN2K)
#define __USE_XOPEN2K
#endif
#endif


#ifdef HAVE_EXECINFO_H
#include <execinfo.h>
#endif


#ifdef __MSC__ 
  #include "ms_stdint.h"
  #include "ms_inttypes.h"
  #include "ms_stat.h"
#else
#ifndef SUNOS
  #include <stdint.h>
#endif
  #include <inttypes.h>
  #include <sys/types.h>
  #include <sys/resource.h>
  #include <sys/time.h>
#endif


/*************************************************************************
* Architecture-specific modifications
**************************************************************************/
#ifdef WIN32
typedef ptrdiff_t ssize_t;
#endif


#ifdef SUNOS
#define PTRDIFF_MAX  INT64_MAX
#endif

#ifdef __MSC__
/* MSC does not have rint() function */
#define rint(x) ((int)((x)+0.5))  

/* MSC does not have INFINITY defined */
#ifndef INFINITY
#define INFINITY FLT_MAX
#endif
#endif

#endif
