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
\file gk_getopt.h
\brief This file contains GNU's externs/structs/prototypes

\date   Started 3/27/2007
\author George
\version\verbatim $Id: gk_getopt.h 10711 2011-08-31 22:23:04Z karypis $ \endverbatim
*/

#ifndef _GK_GETOPT_H_
#define _GK_GETOPT_H_


/* Externals from getopt.c */
extern char *gk_optarg;
extern int gk_optind;
extern int gk_opterr;
extern int gk_optopt;


/*! \brief The structure that stores the information about the command-line options 

This structure describes a single long option name for the sake of 
gk_getopt_long(). The argument <tt>long_options</tt> must be an array 
of these structures, one for each long option. Terminate the array with 
an element containing all zeros.
*/
struct gk_option {
  char *name;       /*!< This field is the name of the option. */
  int has_arg;      /*!< This field says whether the option takes an argument. 
                         It is an integer, and there are three legitimate values: 
                         no_argument, required_argument and optional_argument. 
                         */
  int *flag;        /*!< See the discussion on ::gk_option#val */
  int val;          /*!< These fields control how to report or act on the option 
                         when it occurs. 
                         
                         If flag is a null pointer, then the val is a value which 
                         identifies this option. Often these values are chosen 
                         to uniquely identify particular long options.

                         If flag is not a null pointer, it should be the address 
                         of an int variable which is the flag for this option. 
                         The value in val is the value to store in the flag to 
                         indicate that the option was seen. */
};

/* Names for the values of the `has_arg' field of `struct gk_option'.  */
#define no_argument		0
#define required_argument	1
#define optional_argument	2


/* Function prototypes */
extern int gk_getopt(int __argc, char **__argv, char *__shortopts);
extern int gk_getopt_long(int __argc, char **__argv, char *__shortopts,
              struct gk_option *__longopts, int *__longind);
extern int gk_getopt_long_only (int __argc, char **__argv,
              char *__shortopts, struct gk_option *__longopts, int *__longind);



#endif
