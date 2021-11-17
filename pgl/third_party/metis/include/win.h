/*
\file win32.h
\brief This file contains empty function definition for win32 in METOS so as to solve compiling problem.
*/

#define IDXTYPEWIDTH 64
#define REALTYPEWIDTH 32
#if defined(_MSC_VER)
  #define COMPILER_MSC
#endif
#if defined(__ICC)
  #define COMPILER_ICC
#endif
#if defined(__GNUC__)
  #define COMPILER_GCC
#endif

#ifndef _GKLIB_H_
#ifdef COMPILER_MSC
#include <limits.h>

typedef __int32 int32_t;
typedef __int64 int64_t;
#define PRId32       "I32d"
#define PRId64       "I64d"
#define SCNd32       "ld"
#define SCNd64       "I64d"
#define INT32_MIN    ((int32_t)_I32_MIN)
#define INT32_MAX    _I32_MAX
#define INT64_MIN    ((int64_t)_I64_MIN)
#define INT64_MAX    _I64_MAX
#else
#include <inttypes.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

int METIS_Recursive_win32(int64_t *nvtxs, int64_t *ncon, int64_t *xadj,
                 int64_t *adjncy, int64_t *vwgt, int64_t *vsize, int64_t *adjwgt,
                 int64_t *nparts, float *tpwgts, float *ubvec, int64_t *options,
                 int64_t *edgecut, int64_t *part);

int METIS_Kway_win32(int64_t *nvtxs,int64_t *ncon, int64_t *xadj,
                  int64_t *adjncy, int64_t *vwgt, int64_t *vsize, int64_t *adjwgt,
                  int64_t *nparts, float *tpwgts, float *ubvec, int64_t *options,
                  int64_t *edgecut, int64_t *part);

int METIS_DefaultOptions_win32(int64_t *options);

#ifdef __cplusplus
}
#endif

