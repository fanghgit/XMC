#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include "tron.h"

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern double dnrm2_(int *, double *, int *);
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void TRON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*tron_print_string)(buf);
}

TRON::TRON(const function *fun_obj, double eps, double eps_cg, int max_iter)
{
	this->fun_obj=const_cast<function *>(fun_obj);
	this->eps=eps;
	this->eps_cg=eps_cg;
	this->max_iter=max_iter;
	tron_print_string = default_print;
}

TRON::~TRON()
{
}

void TRON::gd(double *w)
{
	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	double step_size, one=1.0;
	double f, fnew, actred;
	double init_step_size = 1;
	int search = 1, iter = 1, inc = 1;
	double *s = new double[n];
	double *r = new double[n];
	double *g = new double[n];


	double *w0 = new double[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	fun_obj->fun(w0);
	fun_obj->grad(w0, g);
	double gnorm0 = dnrm2_(&n, g, &inc);
	delete [] w0;
	printf("eps = %.16e, |g0| = %.16e\n", eps, gnorm0);
	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	double gnorm = dnrm2_(&n, g, &inc);

	//printf("initial gnorm: %4.8e, f: %4.8e\n",gnorm, f);

	if (gnorm <= eps*gnorm0)
		search = 0;

	iter = 1;
	// calculate gradient norm at w=0 for stopping condition.
	//double *w_new = new double[n];

	//constant stepsize:
	double L = 0.25 + 1.0;
	step_size = 1/L;
	while (iter <= max_iter && search)
	{
		//memcpy(w_new, w, sizeof(double)*n);
		//daxpy_(&n, &one, s, &inc, w_new, &inc);

		//clock_t line_time = clock();

		for(int i=0; i<n; i++)
			w[i] -= step_size*g[i];
			//s[i] = -g[i];
		//step_size = fun_obj->line_search(s, w, g, init_step_size, &fnew);  //fangh comment out


		//printf("stepsize: %1.3e\n", step_size);
		//line_time = clock() - line_time;
		//actred = f - fnew;

		if (step_size == 0)
		{
			info("WARNING: line search fails\n");
			break;
		}
		daxpy_(&n, &step_size, s, &inc, w, &inc);
		//clock_t t = clock();
		//double snorm = dnrm2_(&n, s, &inc);
		//info("iter %2d f %5.10e |g| %5.10e CG %3d step_size %5.3e snorm %5.10e cg_time %f line_time %f time %f \n", iter, f, gnorm, cg_iter, step_size, snorm
		//	,(float(cg_time)/CLOCKS_PER_SEC), (float(line_time)/CLOCKS_PER_SEC), (float(t-start_time))/CLOCKS_PER_SEC);

		f = fnew;
		iter++;

		fun_obj->grad(w, g);

		gnorm = dnrm2_(&n, g, &inc);
		//printf("gnorm: %4.8e, f: %4.8e\n",gnorm, f);
		if (gnorm <= eps*gnorm0)
			break;
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
	}
	printf("num iter: %i\n", iter);
	//printf("time: %f\n", (float(t-start_time))/CLOCKS_PER_SEC );

	delete[] g;
	delete[] r;
	//delete[] w_new;
	delete[] s;
}


void TRON::tron(double *w, clock_t start_time)
{
	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	double step_size, one=1.0;
	double f, fnew, actred;
	double init_step_size = 1;
	int search = 1, iter = 1, inc = 1;
	double *s = new double[n];
	double *r = new double[n];
	double *g = new double[n];

	// calculate gradient norm at w=0 for stopping condition.
	double *w0 = new double[n];
	for (i=0; i<n; i++)
		w0[i] = 0;
	fun_obj->fun(w0);
	fun_obj->grad(w0, g);
	double gnorm0 = dnrm2_(&n, g, &inc);
	delete [] w0;
	printf("eps = %.16e, |g0| = %.16e\n", eps, gnorm0);
	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	double gnorm = dnrm2_(&n, g, &inc);

	if (gnorm <= eps*gnorm0)
		search = 0;

	iter = 1;

	double *w_new = new double[n];
	while (iter <= max_iter && search)
	{
		clock_t cg_time = clock();
		cg_iter = trcg(g, s, r);
		cg_time = clock() - cg_time;

		memcpy(w_new, w, sizeof(double)*n);
		daxpy_(&n, &one, s, &inc, w_new, &inc);

		clock_t line_time = clock();
		step_size = fun_obj->line_search(s, w, g, init_step_size, &fnew);
		line_time = clock() - line_time;
		actred = f - fnew;

		if (step_size == 0)
		{
			info("WARNING: line search fails\n");
			break;
		}
		daxpy_(&n, &step_size, s, &inc, w, &inc);
		clock_t t = clock();
    double snorm = dnrm2_(&n, s, &inc);
		//info("iter %2d f %5.10e |g| %5.10e CG %3d step_size %5.3e snorm %5.10e cg_time %f line_time %f time %f \n", iter, f, gnorm, cg_iter, step_size, snorm
		//	,(float(cg_time)/CLOCKS_PER_SEC), (float(line_time)/CLOCKS_PER_SEC), (float(t-start_time))/CLOCKS_PER_SEC);

		f = fnew;
		iter++;

		fun_obj->grad(w, g);

		gnorm = dnrm2_(&n, g, &inc);
		if (gnorm <= eps*gnorm0)
			break;
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
		/*
		if (fabs(actred) <= 1.0e-12*fabs(f))
		{
			info("WARNING: actred too small\n");
			break;
		}*/
	}
	clock_t t = clock();
	printf("num iter: %i\n", iter);
	//printf("time: %f\n", (float(t-start_time))/CLOCKS_PER_SEC );

	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
}

int TRON::trcg(double *g, double *s, double *r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	double *d = new double[n];
	double *Hd = new double[n];
	double rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = eps_cg*dnrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (dnrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
	delete[] Hd;

	return(cg_iter);
}

double TRON::norm_inf(int n, double *x)
{
	double dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf))
{
	tron_print_string = print_string;
}
