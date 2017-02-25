#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>
#include <string.h>

#include "tv.h"

/*! \file tvmethods.c
 
  \brief Implementation of the algorithm
  
 */

double tv_gsl_f (const gsl_vector *x, void *params);
void tv_gsl_df (const gsl_vector *x, void *params, gsl_vector *df);
void tv_gsl_fdf (const gsl_vector *x, void *params, double *f, gsl_vector *df);

/*! Returns a function to approximate twice the absolute value of x

 */
double tv_psi (double x, void *params)
{
  double beta = *(double *)params;

  return 2 * sqrt (x + beta * beta);
}

/*! Returns a function to approximate the derivative of twice the absolute
    value of x

 */
double tv_dpsi (double x, void *params)
{
  double beta = *(double *)params;

  return 1 / sqrt (x + beta * beta);;
}

/*! Returns a function to approximate the second derivative of twice the absolute
    value of x

 */
double tv_d2psi (double x, void *params)
{
  double beta = *(double *)params;

  return -0.5 * pow (x + beta * beta, -1.5);
}


/*! Allocates the space for the tv method. In this case, n is the dimension of
    the input vector. I don't use the same notation as Vogel, which requires
    the input array to be dimension n + 1

*/
tv * tv_alloc (int n, double dx)
{
  tv *t = malloc (sizeof (tv));
  const gsl_multimin_fdfminimizer_type *T;

  
  /* Scalars */
  t -> n         =  n;
  t -> niter     =  n;
  t -> dx        = dx;
  t -> d0        =  0;
  t -> dmax      =  1;

  t -> alpha     = 0.1;
  t -> beta      = 0.1;

  /* Vectors */
  t -> d      = malloc ((size_t)n * sizeof (double));
  t -> f      = malloc ((size_t)n * sizeof (double));
  t -> d1     = malloc ((size_t)n * sizeof (double));
  
  t -> v1     = malloc ((size_t)n * sizeof (double));
  t -> v2     = malloc ((size_t)n * sizeof (double));
  t -> v3     = malloc ((size_t)n * sizeof (double));

  /* gsl minimization */
  t -> v        = gsl_vector_calloc ((size_t)n);
  t -> F.n      = (size_t)n ;
  t -> F.f      = tv_gsl_f  ;
  t -> F.df     = tv_gsl_df ;
  t -> F.fdf    = tv_gsl_fdf;
  t -> F.params = t         ;

  T = gsl_multimin_fdfminimizer_vector_bfgs2;
  t -> s = gsl_multimin_fdfminimizer_alloc (T, (size_t)n);  

  return t;
}

/*! Cleans the space allocated by a tv_workspace structure

 */
void tv_free (tv *t)
{
  free (t -> d );
  free (t -> f );
  free (t -> d1);

  free (t -> v1);
  free (t -> v2);
  free (t -> v3);

  gsl_vector_free (t -> v);
  gsl_multimin_fdfminimizer_free (t -> s);

  free (t);
  t = NULL;
}

/*! Stores the data in the workspace array. It has to be of dimension n
  
 */
void tv_initdata (double *d, tv *t)
{
  int i;  

  /* Shift */
  t -> d0   = d[0];
  for (i = 0; i < t -> n; i ++) t -> d[i] = d[i] - t -> d0;

  /* Maximum value */
  t -> dmax = 0;
  for (i = 0; i < t -> n; i ++) t -> dmax = MAX(fabs (t -> d[i]), t -> dmax);
  for (i = 0; i < t -> n; i ++) t -> d[i] /= t -> dmax;

  /* Taking the derivative */
  for (i = 1; i < t -> n - 1; i ++) 
    t -> f[i] =  (t -> d[i + 1] - t -> d[i - 1]) / (2 * t -> dx);
  t -> f[0         ] = t -> f[1         ];
  t -> f[t -> n - 1] = t -> f[t -> n - 2];

}

/*! Calculates the i-th entry of the vector Df

 */
double tv_Dif (int i, double *f, tv *t)
{
  double d;

  d = (f[i] - f[i - 1]) / t -> dx;
  
  return d;
}

/*! Calulates the hessian of the penalty functional times a vector: (L'(f)f)g

 */
void tv_Lpfg (double *f, double *g, double *Lpfg, tv *t)
{
  int i;
  double Lp1, Lp2;
  double Dif, Dig;
  
  for (i = 0; i < t -> n; i ++) {
    
    Lp1 = Lp2 = 0;
    if (i != 0) {
      Dif = tv_Dif (i, f, t);
      Dig = tv_Dif (i, g, t);
      Lp1 = 2 * tv_d2psi (Dif * Dif, &t -> beta) * (Dif * Dif) * Dig;      
    }

    if (i != t -> n - 1) {
      Dif = tv_Dif (i + 1, f, t);
      Dig = tv_Dif (i + 1, g, t);
      Lp2 = 2 * tv_d2psi (Dif * Dif, &t -> beta) * (Dif * Dif) * Dig;
    }

    Lpfg[i] = Lp1 - Lp2;
    
  }

}


/*! Calulates the gradient of the penalty functional: L(f)g

 */
void tv_Lfg (double *f, double *g, double *Lf, tv *t)
{
  int i;
  double L1, L2;
  double Dif, Dig;
  
  for (i = 0; i < t -> n; i ++) {
    
    L1 = L2 = 0;
    if (i != 0) {
      Dif = tv_Dif (i, f, t);
      Dig = tv_Dif (i, g, t);
      L1 = tv_dpsi (Dif * Dif, &t -> beta) * Dig;
    }

    if (i != t -> n - 1) {
      Dif = tv_Dif (i + 1, f, t);
      Dig = tv_Dif (i + 1, g, t);
      L2 = tv_dpsi (Dif * Dif, &t -> beta) * Dig;
    }

    Lf [i] = L1 - L2;
  }
  
}

/*! Calculates the vector Kf

 */
void tv_Kf (double *f, double *Kf, tv *t)
{
  int i;
  double s;

  Kf[0] = 0;
  s = 0;
  for (i = 1; i < t -> n; i ++) {
    
    Kf[i] = f[0] + f[i] + 2 * s;
    s += f[i];
    
    Kf[i] *= 0.5 * t -> dx;
  }


}

/*! Calculates the product K^trans f 

 */
void tv_Ktransf (double *f, double *Ktransf, tv *t)
{
  int i;
  double s;

  Ktransf[t -> n - 1] = 0.5 * t -> dx * f[t -> n - 1];
  s = f[t -> n - 1];
  for (i = t -> n - 2; i > 0; i --) {
    Ktransf[i] = f[i] + 2 * s;
    s += f[i];

    Ktransf[i] *= 0.5 * t -> dx;
  }
  Ktransf[0] = 0.5 * t -> dx * s;


}

/* Calculates the gradient of the cost functional 

 */
void tv_gradT (double *f, double *gradT, tv *t)
{
  int i;  

  tv_Kf (f, t -> v1, t);
  for (i = 0; i < t -> n; i ++) t -> v1[i] -= t -> d[i];
  tv_Ktransf (t -> v1, t -> v2, t);

  tv_Lfg (f, f, t -> v1, t);
  for (i = 0; i < t -> n; i ++) gradT[i] = t -> v2[i] + t -> alpha * t -> v1[i];

}

/*! Calculates the function T

 */
double tv_T (double *f, tv *t)
{
  int i;
  double Dif;
  double F = 0, J = 0;

  tv_Kf (f, t -> v1, t);
  for (i = 0; i < t -> n; i ++) t -> v1[i] -= t -> d[i];
  for (i = 0; i < t -> n; i ++) F += (t -> v1[i]) * (t -> v1[i]);
  F *= 0.5;

  for (i = 1; i < t -> n; i ++) {
    Dif = tv_Dif (i, f, t);
    J += tv_psi (Dif * Dif, &t -> beta);
  }
  J *= 0.5 * t -> dx;

  return F + t -> alpha * J;
}

/*! Returns the hessian of the penalty function times a vector H(f)g
   
 */
void tv_hessTfg (double *f, double *g, double *hessTg, tv *t)
{
  int i;

  tv_Lfg  (f, g, t -> v1, t);
  tv_Lpfg (f, g, t -> v2, t);
  for (i = 0; i < t -> n; i ++) t -> v3[i] = t -> v1[i] + t -> v2[i];

  tv_Kf (g, t -> v1, t);
  tv_Ktransf (t -> v1, t -> v2, t);
  for (i = 0; i < t -> n; i ++) hessTg[i] = t -> v2[i] + t -> alpha * t -> v3[i];
}

/*! Provides a function for the gsl minimization algorithm

 */
double tv_gsl_f (const gsl_vector *x, void *params)
{
  tv *t = (tv *)params;
  double *f = x -> data;
  double T = 0;
  
  T = tv_T (f, t);

  return T;
}

/*! Provides a function for the gsl minimization algorithm

 */
void tv_gsl_df (const gsl_vector *x, void *params, gsl_vector *df)
{
  tv *t = (tv *)params;
  double *f = x -> data;
  
  tv_gradT (f, df -> data, t);
}


/*! Provides a function for the gsl minimization algorithm

 */
void tv_gsl_fdf (const gsl_vector *x, void *params, double *f, gsl_vector *df)
{
  *f = tv_gsl_f (x, params);
  tv_gsl_df (x, params, df);
}

/*!  Solves minimization problem

 */
void tv_solve (tv *t)
{
  int i, iter, status;
  const int    maxiter = 100000;
  const double eps     =   1e-6;

  /* Initializing */
  memcpy (t -> v -> data, t -> f, (size_t)t -> n * sizeof (double)); 
  
  
  gsl_multimin_fdfminimizer_set (t -> s, &(t -> F), t -> v, 0.1, 0.1);
  iter = 0;

  /* Iterating */  
  do {

    iter ++;
    status = gsl_multimin_fdfminimizer_iterate (t -> s);

    if (status) break;
    status = gsl_multimin_test_gradient (t -> s -> gradient, eps);


  } while (status == GSL_CONTINUE && iter < maxiter);  

   
  /* Storing */
  gsl_vector_scale (t -> s -> x, t -> dmax);
  memcpy (t -> f, t -> s -> x -> data, (size_t)t -> n * sizeof (double)); 
  t -> niter = iter;

  /* Data */
  tv_Kf (t -> f, t -> d1, t);
  for (i = 0; i < t -> n; i ++) t -> d1[i] += t -> d0;

}
