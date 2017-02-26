#include <Python.h>
#include <numpy/arrayobject.h>
#include "tv.h"

/*! \file tvmodule.c
 
  \brief Interface for python
  
 */


static PyObject * py_tv (PyObject *self, PyObject *args, PyObject *keywds)
{
  double alpha;
  double dx;
  double *data, *diff;
  int n;

  static char *kwlist[] = {"data", "deriv", "alpha", "dx", NULL}; 
  PyObject *data_obj, *data_np;
  PyObject *diff_obj, *diff_np;

  tv *t;  

  /* Default parameters */
  alpha = 0.1;
  dx = 1.;

  /* Parsing input */
  if (!PyArg_ParseTupleAndKeywords (args, keywds, "OO|dd", kwlist, &data_obj, &diff_obj, &alpha, &dx))
    return NULL;
  
  /* Casting object as numpy array */
  data_np = PyArray_FROM_OTF (data_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  diff_np = PyArray_FROM_OTF (diff_obj, NPY_DOUBLE, NPY_IN_ARRAY);
  if (data_np == NULL || diff_np == NULL) {
    Py_XDECREF (data_np);
    return NULL;
  }
  
  /* Casting to C array */
  n = (int) PyArray_DIM (data_np, 0);
  data = (double*)PyArray_DATA (data_np);
  diff = (double*)PyArray_DATA (diff_np);

  /* Allocating tv */
  t = tv_alloc (n, dx);
  t -> alpha  = alpha;

  /* Solving */
  tv_initdata (data, t);
  tv_solve (t);
  memcpy (diff, t -> f, (size_t)n * sizeof (double));

  /* Cleaning */
  tv_free (t);
  Py_DECREF (data_np);
  
  self = self;
  Py_RETURN_NONE;
}

static PyMethodDef TVMethods[] =
  {
    {"tv", (PyCFunction)py_tv, METH_VARARGS | METH_KEYWORDS, "Calculates the first derivative using tv"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC inittv (void)
{
  (void) Py_InitModule ("tv", TVMethods);

  import_array ();
}

