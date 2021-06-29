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

#ifndef _MSC_VER // [
#error "Use this header only with Microsoft Visual C++ compilers!"
#endif // _MSC_VER ]

#ifndef _MS_STAT_H_
#define _MS_STAT_H_

#if _MSC_VER > 1000
#pragma once
#endif

#include <sys/stat.h>
/* Test macros for file types.  */

#define __S_ISTYPE(mode, mask)  (((mode) & S_IFMT) == (mask))

#define S_ISDIR(mode)    __S_ISTYPE((mode), S_IFDIR)
#define S_ISCHR(mode)    __S_ISTYPE((mode), S_IFCHR)
#define S_ISBLK(mode)    __S_ISTYPE((mode), S_IFBLK)
#define S_ISREG(mode)    __S_ISTYPE((mode), S_IFREG)

#endif 
