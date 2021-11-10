/*
\file win32.h
\brief This file contains empty function definition for win32 in METOS so as to solve compiling problem.
*/

/*---------------------------------------------------------------------

* Setup the basic datatypes
*----------------------------------------------------------------------*/
#if IDXTYPEWIDTH == 32
  typedef int32_t idx_t;

  #define IDX_MAX   INT32_MAX
  #define IDX_MIN   INT32_MIN

  #define SCIDX  SCNd32
  #define PRIDX  PRId32

  #define strtoidx      strtol
  #define iabs          abs
#elif IDXTYPEWIDTH == 64
  typedef int64_t idx_t;

  #define IDX_MAX   INT64_MAX
  #define IDX_MIN   INT64_MIN

  #define SCIDX  SCNd64
  #define PRIDX  PRId64

#ifdef COMPILER_MSC
  #define strtoidx      _strtoi64
#else
  #define strtoidx      strtoll
#endif
  #define iabs          labs
#else
  #error "Incorrect user-supplied value fo IDXTYPEWIDTH"
#endif


#if REALTYPEWIDTH == 32
  typedef float real_t;

  #define SCREAL         "f"
  #define PRREAL         "f"
  #define REAL_MAX       FLT_MAX
  #define REAL_MIN       FLT_MIN
  #define REAL_EPSILON   FLT_EPSILON

  #define rabs          fabsf
  #define REALEQ(x,y) ((rabs((x)-(y)) <= FLT_EPSILON))

#ifdef COMPILER_MSC
  #define strtoreal     (float)strtod
#else
  #define strtoreal     strtof
#endif
#elif REALTYPEWIDTH == 64
  typedef double real_t;

  #define SCREAL         "lf"
  #define PRREAL         "lf"
  #define REAL_MAX       DBL_MAX
  #define REAL_MIN       DBL_MIN
  #define REAL_EPSILON   DBL_EPSILON

  #define rabs          fabs
  #define REALEQ(x,y) ((rabs((x)-(y)) <= DBL_EPSILON))

  #define strtoreal     strtod
#else
  #error "Incorrect user-supplied value for REALTYPEWIDTH"
#endif


/*------------------------------------------------------------------------
* Function prototypes
*-------------------------------------------------------------------------*/

#ifdef _WINDLL
#define METIS_API(type) __declspec(dllexport) type __cdecl
#elif defined(__cdecl)
#define METIS_API(type) type __cdecl
#else
#define METIS_API(type) type
#endif


#ifdef __cplusplus
extern "C" {
#endif
int METIS_Recursive_win32(idx_t *nvtxs, idx_t *ncon, idx_t *xadj,
                  idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt,
                  idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options,
                  idx_t *edgecut, idx_t *part) {
  return 1;
}
int METIS_Kway_win32(idx_t *nvtxs, idx_t *ncon, idx_t *xadj,
                  idx_t *adjncy, idx_t *vwgt, idx_t *vsize, idx_t *adjwgt,
                  idx_t *nparts, real_t *tpwgts, real_t *ubvec, idx_t *options,
                  idx_t *edgecut, idx_t *part) {
  return 1;
}
int METIS_DefaultOptions_win32(idx_t *options) {
  return 1;
}
}

#ifdef __cplusplus
}
#endif
