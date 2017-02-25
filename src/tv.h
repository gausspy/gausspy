/*! \file tv.h
 
  \brief This file cotains all macros and global variables used along
  the program
  
 */

#ifndef TV_H
#define ALLVARS_H

#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

#ifndef CONCAT2
#define CONCAT2x(a,b) a ## _ ## b    /*!< Merge two args */
#define CONCAT2(a,b) CONCAT2x(a,b)   /*!< Merge two args */
#endif
#define MAX(x, y) ((x) > (y) ? (x) : (y)) /*!< Maximum between two numbers */
#define MIN(x, y) ((x) < (y) ? (x) : (y)) /*!< Minimum between two numbers */

typedef enum {                       /*!< True/False boolen variable  */
  false,                             /*!< False */
  true                               /*!< True */
} boolean; 

typedef struct {                     /*!< TV workspace */
  
  int n;                             /*!< problem size */
  double dx;                         /*!< dx */
  double d0;                         /*!< First entry of the data array */
  double dmax;                       /*!< Maximum value of |data| */

  double alpha;                      /*!< alpha */
  double beta;                       /*!< beta */

  double *d;                         /*!< data                */
  double *f;                         /*!< derivative          */  
  double *d1;                        /*!< approximation to the data */

  double *v1;                        /*!< tmp vector */
  double *v2;                        /*!< tmp vector */
  double *v3;                        /*!< tmp vector */
  
  int niter;                         /*!< Number of iterations */

  gsl_vector *v;                     /*!< tmp vector for evaluating the minimization  */
  gsl_multimin_function_fdf F;       /*!< gsl function to evaluate T and its derivative */
  gsl_multimin_fdfminimizer *s;      /*!< minimizer for the line-search */


} tv;

tv * tv_alloc (int n, double dx);
void tv_free (tv *t);
void tv_initdata (double *d, tv *t);
void tv_solve (tv *t);

#endif


