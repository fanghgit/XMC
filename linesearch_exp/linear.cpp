#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include <time.h>
#include <limits>
#include <vector>
#include "linear.h"
#include "tron.h"
#include <queue>
#include <list>
#include <stack>
#include <unordered_map>
#include <algorithm>
#include <omp.h>



typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void print_null(const char *s) {}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

// define MST problem
class label_node
{
public:
	int id;
	double *w;  //primal parameters
	double *alpha; // dual parameters
	bool visited;
	//bool isparent;
	std::vector<int> neighbours;
	std::vector<int> children;
	int parent;
	label_node() {
		this->id = -1;
		this->w = NULL;
		this->alpha = NULL;
		this->visited = false;
		//this->isparent = false;
		this->parent = -1;
		this->neighbours.resize(0);
		this->children.resize(0);
	}
};
//
// void label_node::add_child(label_node *child)
// {
// 	this->children.push_back(child);
// }

class Graph{
private:
	int V, E;
	std::vector<std::pair<int, std::pair<int, int> > > edges;
public:
	std::vector<std::pair<int, std::pair<int, int> > > MST;
	Graph(int V, int E);
	void addEdge(int u, int v, int w);
	int kruskalMST();
	void printMST();
};

Graph::Graph(int V, int E)
{
	this->V = V;
	this->E = E;
}

void Graph::addEdge(int u, int v, int w)
{
	edges.push_back( std::make_pair(w, std::make_pair(u,v) ) );
}

void Graph::printMST(){
    std::vector<std::pair<int,std::pair<int,int> > >::iterator it;
		//auto it;
		for(it = MST.begin();it!=MST.end();it++){
        //cout << it->second.first << " - " << it->second.second << endl;
				printf("%d - %d", it->second.first,it->second.second);
    }
}

struct DisjointSet{
    int *parent,*rnk;
    int n;

    DisjointSet(int n){
        this->n = n;
        parent = new int[n+1];
        rnk = new int[n+1];

        for(int i=0;i<=n;i++){
            rnk[i] = 0;
            parent[i] = i;
        }
    }
    int Find(int u){
        if(u != parent[u])
					parent[u] = Find(parent[u]);
        return parent[u];
    }

    void Union(int x,int y){
        // x = Find(x);
        // y = Find(y);
        // if(x != y){
        //     if(rnk[x] < rnk[y]){
        //         rnk[y] += rnk[x];
        //         parent[x] = y;
        //     }
        //     else{
        //         rnk[x] += rnk[y];
        //         parent[y] = x;
        //     }
        // }
				x = Find(x), y = Find(y);

        /* Make tree with smaller height
           a subtree of the other tree  */
        if (rnk[x] > rnk[y])
            parent[y] = x;
        else // If rnk[x] <= rnk[y]
            parent[x] = y;

        if (rnk[x] == rnk[y])
            rnk[y]++;
    }
};

bool comp(const std::pair<int, std::pair<int,int> > &lhs, const std::pair<int, std::pair<int,int> > &rhs)
{
	if(lhs.first < rhs.first)
		return true;
	else if((lhs.first == rhs.first) && (lhs.second.first == 0 || lhs.second.second == 0)) // give priority to label 0
		return true;
	else
		return false;
}

int Graph::kruskalMST(){
    int MSTWeight = 0; //sum of all vertex weights
    std::sort(edges.begin(),edges.end(), comp);
		//printf("|E| in kruskal: %d\n", edges.size());
    //for all u in G_v
    //    MAKE-SET(u)
    DisjointSet ds(this->V);

    std::vector<std::pair<int,std::pair<int,int> > >::iterator it;
		//auto it;
		// for all edges in G
    for(it = edges.begin(); it!=edges.end();it++){
        int u = it->second.first;
        int v = it->second.second;

        int setU = ds.Find(u);
        int setV = ds.Find(v);


        if(setU != setV){
            int w = it->first;
            MST.push_back( std::make_pair(w, std::make_pair(u,v) ) );
            MSTWeight += it->first;

            ds.Union(setU,setV);
        }
    }
    return MSTWeight;
}


class sparse_operator
{
public:
	static double nrm2_sq(const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += x->value*x->value;
			x++;
		}
		return (ret);
	}

	static double dot(const double *s, const feature_node *x)
	{
		double ret = 0;
		while(x->index != -1)
		{
			ret += s[x->index-1]*x->value;
			x++;
		}
		return (ret);
	}

	static void axpy(const double a, const feature_node *x, double *y)
	{
		while(x->index != -1)
		{
			y[x->index-1] += a*x->value;
			x++;
		}
	}
};


class l2r_erm_fun: public function
{
public:
	l2r_erm_fun(const subproblem *prob, double *C);
	~l2r_erm_fun();

	double fun(double *w);
	double line_search(double *d, double *w, double *g, double alpha, double *f);
	int get_nr_variable(void);

protected:
	virtual double C_times_loss(int i, double wx_i) = 0;
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	const subproblem *prob;
	double *wx;
	double *tmp;
	double wTw;
	double current_f;
};

l2r_erm_fun::l2r_erm_fun(const subproblem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	wx = new double[l];
	tmp = new double[l];
	this->C = C;
}

l2r_erm_fun::~l2r_erm_fun()
{
	delete[] wx;
	delete[] tmp;
}

double l2r_erm_fun::fun(double *w)
{
	int i;
	double f=0;
	int l=prob->l;
	int w_size=get_nr_variable();

	wTw = 0;
	Xv(w, wx);

	for(i=0;i<w_size;i++)
		wTw += w[i]*w[i];
	for(i=0;i<l;i++)
		f += C_times_loss(i, wx[i]);
	f = f + 0.5 * wTw;

	current_f = f;
	return(f);
}

int l2r_erm_fun::get_nr_variable(void)
{
	return prob->n;
}

double l2r_erm_fun::line_search(double *d, double *w, double *g, double alpha, double *f)
{
	int i;
	int l = prob->l;
	double dTd = 0;
	double wTd = 0;
	double gTd = 0;
	double eta = 0.01;
	int w_size = get_nr_variable();
	int max_num_linesearch = 1000;
	Xv(d, tmp);

	for (i=0;i<w_size;i++)
	{
		dTd += d[i] * d[i];
		wTd += d[i] * w[i];
		gTd += d[i] * g[i];
	}
	int num_linesearch = 0;
	for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
	{
		double loss = 0;
		for(i=0;i<l;i++)
		{
			double inner_product = tmp[i] * alpha + wx[i];
			loss += C_times_loss(i, inner_product);
		}
		*f = loss + (alpha * alpha * dTd + wTw) / 2.0 + alpha * wTd;
		if (*f - current_f <= eta * alpha * gTd)
		{
			for (i=0;i<l;i++)
				wx[i] += alpha * tmp[i];
			break;
		}
		else
			alpha *= 0.5;
	}

	if (num_linesearch >= max_num_linesearch)
	{
		*f = current_f;
		return 0;
	}

	wTw += alpha * alpha * dTd + 2* alpha * wTd;
	current_f = *f;
	return alpha;
}

void l2r_erm_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
		Xv[i]=sparse_operator::dot(v, x[i]);
}

void l2r_erm_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
		sparse_operator::axpy(v[i], x[i], XTv);
}

class l2r_lr_fun: public l2r_erm_fun
{
public:
	l2r_lr_fun(const subproblem *prob, double *C);
	~l2r_lr_fun();

	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

private:
	double *D;
	double C_times_loss(int i, double wx_i);
};

l2r_lr_fun::l2r_lr_fun(const subproblem *prob, double *C):
	l2r_erm_fun(prob, C)
{
	int l=prob->l;
	D = new double[l];
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] D;
}

double l2r_lr_fun::C_times_loss(int i, double wx_i)
{
	double ywx_i = wx_i * prob->y[i];
	if (ywx_i >= 0)
		return C[i]*log(1 + exp(-ywx_i));
	else
		return C[i]*(-ywx_i + log(1 + exp(ywx_i)));
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		tmp[i] = 1/(1 + exp(-y[i]*wx[i]));
		D[i] = tmp[i]*(1-tmp[i]);
		tmp[i] = C[i]*(tmp[i]-1)*y[i];
	}
	XTv(tmp, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + g[i];
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i] = 0;
	for(i=0;i<l;i++)
	{
		feature_node * const xi=x[i];
		wa[i] = sparse_operator::dot(s, xi);

		wa[i] = C[i]*D[i]*wa[i];

		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

class l2r_l2_svc_fun: public l2r_erm_fun
{
public:
	l2r_l2_svc_fun(const subproblem *prob, double *C);
	~l2r_l2_svc_fun();

	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

protected:
	void subXTv(double *v, double *XTv);

	int *I;
	int sizeI;

private:
	double C_times_loss(int i, double wx_i);
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const subproblem *prob, double *C):
	l2r_erm_fun(prob, C)
{
	I = new int[prob->l];
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] I;
}

double l2r_l2_svc_fun::C_times_loss(int i, double wx_i)
{
		double d = 1 - prob->y[i] * wx_i;
		if (d > 0)
			return C[i] * d * d;
		else
			return 0;
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
	{
		tmp[i] = wx[i] * y[i];
		if (tmp[i] < 1)
		{
			tmp[sizeI] = C[i]*y[i]*(tmp[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	}
	subXTv(tmp, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		Hs[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node * const xi=x[I[i]];
		wa[i] = sparse_operator::dot(s, xi);

		wa[i] = C[I[i]]*wa[i];

		sparse_operator::axpy(wa[i], xi, Hs);
	}
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
		sparse_operator::axpy(v[i], x[I[i]], XTv);
}


// A coordinate descent algorithm for
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_l2_svc(
	subproblem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int max_iter = 1000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}
	for(j=0; j<w_size; j++)
	{
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x->value *= y[ind]; // x->value stores yi*xij
			double val = x->value;
			b[ind] -= w[j]*val;
			xj_sq[j] += C[GETI(ind)]*val*val;
			x++;
		}
	}

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[GETI(ind)]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*w[j])
				d = -Gp/H;
			else if(Gn > H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					sparse_operator::axpy(d_diff, x, b);
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						if(b[ind] > 0)
							loss_old += C[GETI(ind)]*b[ind]*b[ind];
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					sparse_operator::axpy(-w[i], x, b);
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[GETI(j)]*b[j]*b[j];

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}


// A coordinate descent algorithm for
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr(
	const subproblem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, newton_iter=0, iter=0;
	int max_newton_iter = 100;
	int max_iter = 1000;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double w_norm, w_norm_new;
	double z, G, H;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = INF;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *Hdiag = new double[w_size];
	double *Grad = new double[w_size];
	double *wpd = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xTd = new double[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *tau = new double[l];
	double *D = new double[l];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;

		exp_wTx[j] = 0;
	}

	w_norm = 0;
	for(j=0; j<w_size; j++)
	{
		w_norm += fabs(w[j]);
		wpd[j] = w[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			double val = x->value;
			exp_wTx[ind] += w[j]*val;
			if(y[ind] == -1)
				xjneg_sum[j] += C[GETI(ind)]*val;
			x++;
		}
	}
	for(j=0; j<l; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C[GETI(j)]*tau_tmp;
		D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			Hdiag[j] = nu;
			Grad[j] = 0;

			double tmp = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				Hdiag[j] += x->value*x->value*D[ind];
				tmp += x->value*tau[ind];
				x++;
			}
			Grad[j] = -tmp + xjneg_sum[j];

			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(j=0; j<QP_active_size; j++)
			{
				int i = j+rand()%(QP_active_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<QP_active_size; s++)
			{
				j = index[s];
				H = Hdiag[j];

				x = prob_col->x[j];
				G = Grad[j] + (wpd[j]-w[j])*nu;
				while(x->index != -1)
				{
					int ind = x->index-1;
					G += x->value*D[ind]*xTd[ind];
					x++;
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(wpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[j])
					z = -Gp/H;
				else if(Gn > H*wpd[j])
					z = -Gn/H;
				else
					z = -wpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[j] += z;

				x = prob_col->x[j];
				sparse_operator::axpy(z, x, xTd);
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		if(iter >= max_iter)
			info("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		w_norm_new = 0;
		for(j=0; j<w_size; j++)
		{
			delta += Grad[j]*(wpd[j]-w[j]);
			if(wpd[j] != 0)
				w_norm_new += fabs(wpd[j]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(int i=0; i<l; i++)
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<l; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(j=0; j<w_size; j++)
					w[j] = wpd[j];
				for(int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D[i] = C[GETI(i)]*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(j=0; j<w_size; j++)
				{
					wpd[j] = (w[j]+wpd[j])*0.5;
					if(wpd[j] != 0)
						w_norm_new += fabs(wpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				sparse_operator::axpy(w[i], x, exp_wTx);
			}

			for(int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;

		//info("iter %3d  #CD cycles %d\n", newton_iter, iter);
	}

	info("=========================\n");
	info("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		info("WARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	for(j=0; j<l; j++)
		if(y[j] == 1)
			v += C[GETI(j)]*log(1+1/exp_wTx[j]);
		else
			v += C[GETI(j)]*log(1+exp_wTx[j]);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] Hdiag;
	delete [] Grad;
	delete [] wpd;
	delete [] xjneg_sum;
	delete [] xTd;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] tau;
	delete [] D;
}

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 3 of Hsieh et al., ICML 2008

// #undef GETI
// #define GETI(i) (y[i]+1)
// // To support weights for instances, use GETI(i) (i)
//
// static void solve_l2r_l1l2_svc(
// 	const subproblem *prob, double *w, double *alpha, double eps,
// 	double Cp, double Cn, int solver_type)
// {
// 	int l = prob->l;
// 	int w_size = prob->n;
// 	int i, s, iter = 0;
// 	double C, d, G;
// 	double *QD = new double[l];
// 	int max_iter = 1000;
// 	int *index = new int[l];
// 	//double *alpha = new double[l];
// 	schar *y = new schar[l];
// 	int active_size = l;
//
// 	// PG: projected gradient, for shrinking and stopping
// 	double PG;
// 	double PGmax_old = INF;
// 	double PGmin_old = -INF;
// 	double PGmax_new, PGmin_new;
//
// 	// default solver_type: L2R_L2LOSS_SVC_DUAL
// 	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
// 	double upper_bound[3] = {INF, 0, INF};
// 	if(solver_type == L2R_L1LOSS_SVC_DUAL)
// 	{
// 		diag[0] = 0;
// 		diag[2] = 0;
// 		upper_bound[0] = Cn;
// 		upper_bound[2] = Cp;
// 	}
//
// 	std::vector<int> active_set (0);
// 	for(i=0; i<l; i++)
// 	{
// 		if(prob->y[i] > 0)
// 		{
// 			y[i] = +1;
// 			active_set.push_back(i);
// 		}
// 		else
// 		{
// 			y[i] = -1;
// 		}
// 	}
//
// 	// Initial alpha can be set here. Note that
// 	// 0 <= alpha[i] <= upper_bound[GETI(i)]
// 	//for(i=0; i<l; i++)
// 	//	alpha[i] = 0;
//
// 	for(i=0; i<w_size; i++)
// 		w[i] = 0;
// 	for(i=0; i<l; i++)
// 	{
// 		QD[i] = diag[GETI(i)];
//
// 		feature_node * const xi = prob->x[i];
// 		QD[i] += sparse_operator::nrm2_sq(xi);
// 		sparse_operator::axpy(y[i]*alpha[i], xi, w);
//
// 		index[i] = i;
// 	}
//
// 	// initial change
// 	for(s=0; s<active_set.size(); s++)
// 	{
// 		i = index[s];
// 		const schar yi = y[i];
// 		feature_node * const xi = prob->x[i];
//
// 		G = yi*sparse_operator::dot(w, xi)-1;
//
// 		C = upper_bound[GETI(i)];
// 		G += alpha[i]*diag[GETI(i)];
//
// 		if (alpha[i] == 0)
// 		{
// 			PG = min(G, 0.0);
// 		}
// 		else if(alpha[i] == C)
// 		{
// 			PG = max(G, 0.0);
// 		}
// 		else
// 			PG = G;
//
// 		if(fabs(PG) > 1.0e-12)
// 		{
// 			double alpha_old = alpha[i];
// 			alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
// 			d = (alpha[i] - alpha_old)*yi;
// 			sparse_operator::axpy(d, xi, w);
// 		}
//
// 	}
//
//
// 	printf("Initial*************\n\n");
//
// 	double v = 0;
// 	int nSV = 0;
// 	for(i=0; i<w_size; i++)
// 		v += w[i]*w[i];
// 	for(i=0; i<l; i++)
// 	{
// 		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
// 		if(alpha[i] > 0)
// 			++nSV;
// 	}
// 	info("Initial objective value = %lf\n",v/2);
// 	info("nSV = %d\n",nSV);
//
// 	printf("\n\n***************");
//
// 	while (iter < max_iter)
// 	{
// 		PGmax_new = -INF;
// 		PGmin_new = INF;
//
// 		for (i=0; i<active_size; i++)
// 		{
// 			int j = i+rand()%(active_size-i);
// 			swap(index[i], index[j]);
// 		}
//
// 		for (s=0; s<active_size; s++)
// 		{
// 			i = index[s];
// 			const schar yi = y[i];
// 			feature_node * const xi = prob->x[i];
//
// 			G = yi*sparse_operator::dot(w, xi)-1;
//
// 			C = upper_bound[GETI(i)];
// 			G += alpha[i]*diag[GETI(i)];
//
// 			PG = 0;
// 			if (alpha[i] == 0)
// 			{
// 				if (G > PGmax_old)
// 				{
// 					active_size--;
// 					swap(index[s], index[active_size]);
// 					s--;
// 					continue;
// 				}
// 				else if (G < 0)
// 					PG = G;
// 			}
// 			else if (alpha[i] == C)
// 			{
// 				if (G < PGmin_old)
// 				{
// 					active_size--;
// 					swap(index[s], index[active_size]);
// 					s--;
// 					continue;
// 				}
// 				else if (G > 0)
// 					PG = G;
// 			}
// 			else
// 				PG = G;
//
// 			PGmax_new = max(PGmax_new, PG);
// 			PGmin_new = min(PGmin_new, PG);
//
// 			if(fabs(PG) > 1.0e-12)
// 			{
// 				double alpha_old = alpha[i];
// 				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
// 				d = (alpha[i] - alpha_old)*yi;
// 				sparse_operator::axpy(d, xi, w);
// 			}
// 		}
//
// 		iter++;
// 		if(iter % 10 == 0)
// 			info(".");
//
// 		if(PGmax_new - PGmin_new <= eps)
// 		{
// 			if(active_size == l)
// 				break;
// 			else
// 			{
// 				active_size = l;
// 				info("*");
// 				PGmax_old = INF;
// 				PGmin_old = -INF;
// 				continue;
// 			}
// 		}
// 		PGmax_old = PGmax_new;
// 		PGmin_old = PGmin_new;
// 		if (PGmax_old <= 0)
// 			PGmax_old = INF;
// 		if (PGmin_old >= 0)
// 			PGmin_old = -INF;
// 	}
//
// 	info("\noptimization finished, #iter = %d\n",iter);
// 	if (iter >= max_iter)
// 		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");
//
// 	// calculate objective value
//
// 	v = 0;
// 	nSV = 0;
// 	for(i=0; i<w_size; i++)
// 		v += w[i]*w[i];
// 	for(i=0; i<l; i++)
// 	{
// 		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
// 		if(alpha[i] > 0)
// 			++nSV;
// 	}
// 	info("Objective value = %lf\n",v/2);
// 	info("nSV = %d\n",nSV);
//
// 	delete [] QD;
// 	//delete [] alpha;
// 	delete [] y;
// 	delete [] index;
// }


// transpose matrix X from row format to column format
static void transpose(const subproblem *prob, feature_node **x_space_ret, subproblem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}


// solve MST problem
void order_schedule(const problem *prob, const parameter *param, int nr_class, label_node* nodes)
{
	// dense sheduling
	int l = prob->l;
	std::vector<std::vector<double> > dist_mat (nr_class);


	int l = prob->l;
	int num_all_labels = 0;
	int nnz_upper_bound = 0;
	for(int i=0; i<l; i++)
	{
		num_all_labels += prob->numLabels[i];
		nnz_upper_bound += prob->numLabels[i] * prob->numLabels[i];
	}
	printf("naive upper bound: %d\n", num_all_labels);
	printf("nnz upper bound: %d\n", nnz_upper_bound);

	// brute force
	//std::unordered_map< pair<int, int>, int > hashmap;
	std::vector<std::unordered_map<int, int> > dist_mat (nr_class);
	for(int i=0; i<l; i++)
	{
		for(int j=0; j<prob->numLabels[i]; j++)
		{
			for(int k=j+1; k<prob->numLabels[i]; k++)
			{
				int lbl1 = prob->y[i][j];
				int lbl2 = prob->y[i][k];
				std::unordered_map<int,int>::iterator got = dist_mat[lbl1-1].find (lbl2);
				if(got == dist_mat[lbl1-1].end() )
					dist_mat[lbl1-1][lbl2] = 1;
				else
					dist_mat[lbl1-1][lbl2] += 1;

				got = dist_mat[lbl2-1].find (lbl1);
				if(got == dist_mat[lbl2-1].end() )
					dist_mat[lbl2-1][lbl1] = 1;
				else
					dist_mat[lbl2-1][lbl1] += 1;
			}
		}
	}


  // count n_pos per label;
	std::vector<int> num_pos_per_label (nr_class);
	for(int i=0; i<nr_class; i++)
		num_pos_per_label[i] = 0;

	for(int i=0; i<prob->l; i++)
	{
		for(int j=0; j<prob->numLabels[i]; j++)
		{
			num_pos_per_label[ prob->y[i][j]-1 ]++;
		}
	}

	//construct distance matrix
	double Ecount = 0;
	for(int i=0; i<nr_class; i++)
	{
		std::unordered_map<int,int>::iterator it;
		for(it=dist_mat[i].begin(); it != dist_mat[i].end(); it++)
		{
			int lbl1 = i+1;
			int lbl2 = it->first;
			it->second = num_pos_per_label[lbl1-1] + num_pos_per_label[lbl2-1] - 2*it->second;

			Ecount += 0.5;
		}
		dist_mat[i][0] = num_pos_per_label[i];
		Ecount += 1;
	}
	int E = (int) Ecount;


	//construct ordered distance vector
	// vector< graph_edge > dist_vec;
	// for(int i=0; i<nr_class; i++)
	// {
	// 	std::unordered_map<int,double>::iterator it;
	// 	for(it=dist_mat[i].begin(); it != dist_mat[i].end(); it++){
	// 		if( (i+1) > it->first )
	// 		{
	// 			struct graph_edge tmp;
	// 			tmp.node1 = i+1;
	// 			tmp.node2 = it->first;
	// 			tmp.weight = it->second;
	// 			dist_vec.push_back( tmp );
	// 		}
	// 	}
	// }
	int V = nr_class + 1;
	printf("before construct g, |E|: %d\n", E);
	Graph g(V, E);
	int u,v,w;
	for(int i=0; i<nr_class; i++)
	{
		std::unordered_map<int,int>::iterator it;
		for(it=dist_mat[i].begin(); it != dist_mat[i].end(); it++)
		{
			if( (i+1) > it->first )
			{
				g.addEdge( i+1, it->first, it->second );
				//if(it->second == 0)
					//printf("label %d and label %d are the same!\n");
			}
		}
	}
	printf("adding edge complete\n");

	int weight = g.kruskalMST();
	printf("weight of MST is: %d\n", weight);


	// construct label_node
	std::vector<std::pair<int,std::pair<int,int> > >::iterator it;
	for(it = g.MST.begin();it!=g.MST.end();it++){
		int lbl1 = it->second.first;
		int lbl2 = it->second.second;
		nodes[lbl1].neighbours.push_back(lbl2);
		nodes[lbl2].neighbours.push_back(lbl1);
	}

	//dfs traversal, calculate height and store parents and children
	int height;
	int node_idx = 0;
	std::stack<int> s;
	s.push(node_idx);
	while(!s.empty())
	{
		int popped = s.top();
		nodes[popped].visited = true;
		s.pop();
		for(int ii=0; ii<nodes[popped].neighbours.size(); ii++)
		{
			int cc = nodes[popped].neighbours[ii];
			if(!nodes[cc].visited)
			{
				nodes[popped].children.push_back(cc);
				nodes[cc].parent = popped;
				s.push(cc);
			}
		}
	}

}

// void bfs(label_node* nodes, int start_node, int nr_class, std::vector<std::pair<int, int> > &res)
// {
// 	bool visited[nr_class+1];
// 	for(int i=0; i<(nr_class+1); i++)
// 		visited[i] = false;
//
// 	visited[start_node] = true;
// 	std::list<int> q;
// 	q.push_back(start_node);
//
// 	while(!q.empty())
// 	{
// 		int s = q.front();
// 		q.pop_front();
// 		for(int i=0; i<nodes[s].neighbours.size();i++)
// 		{
// 			if(!visited[nodes[s].neighbours[i] ] )
// 			{
// 				nodes[s].isparent = true;
// 				res.push_back( std::make_pair(s, nodes[s].neighbours[i] ) );
// 				visited[nodes[s].neighbours[i] ] = true;
// 				q.push_back( nodes[s].neighbours[i] );
// 			}
// 		}
// 	}
//
// }


static void train_one(const subproblem *prob, const parameter *param, double *w, double *alpha, double Cp, double Cn)
{
	double wtime = omp_get_wtime();
	clock_t start_time = clock();
	//inner and outer tolerances for TRON
	double eps = param->eps;
	double eps2 = param->eps2;
	double eps_cg = 0.1;
	if(param->init_sol != NULL)
		eps_cg = 0.5;

	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i] > 0)
			pos++;
	neg = prob->l - pos;
	printf("|pos|: %d\n", pos);
	double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;
	primal_solver_tol = min(primal_solver_tol, eps2);
	//double primal_solver_tol = eps;

	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		// case L2R_L2LOSS_SVC_DUAL:
		// 	solve_l2r_l1l2_svc(prob, w, alpha, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
		// 	break;
		// case L2R_L1LOSS_SVC_DUAL:
		// 	solve_l2r_l1l2_svc(prob, w, alpha, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
		// 	break;
		case L2R_LR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w, start_time);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w, start_time);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC_GD:
		{
			printf("using GD\n");
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.gd(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L1R_L2LOSS_SVC:
		{
			subproblem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_l2_svc(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L1R_LR:
		{
			subproblem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
	}

	printf("time spent on train one: %lf\n", omp_get_wtime() - wtime);
}


void dfs(model *model_, const problem *prob, const parameter *param, label_node* nodes, int *classCount, int **labelInd, double *weighted_C, int node_idx, int nr_class)
{
	nodes[node_idx].visited = true;  // writing to different node in different threads, should have no conflicts
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	double *alpha = NULL;  // initialize alpha

	// solve to subproblem for label "node_idx"

	int parent = nodes[node_idx].parent;
	int child = node_idx-1;

	subproblem sub_prob_omp;
	sub_prob_omp.l = l;
	sub_prob_omp.n = n;
	sub_prob_omp.x = prob->x;
	sub_prob_omp.y = Malloc(double,l);

	for(int k=0; k <l; k ++){
		sub_prob_omp.y[k] = -1;
	}

	int jj;
	for(jj=0; jj < classCount[child]; jj++){
		int ind = labelInd[child][jj];
		sub_prob_omp.y[ind] = +1;
	}


	if(nodes[child+1].children.size() > 0)
	{
		nodes[child+1].w = Malloc(double, w_size);
		// if(param->solver_type == L2R_L2LOSS_SVC_DUAL || param->solver_type == L2R_L1LOSS_SVC_DUAL)
		// 	nodes[child+1].alpha = Malloc(double, l);
	}
	// initialize w
	double *w=Malloc(double, w_size);
	// initialize alpha
	// if(param->solver_type == L2R_L2LOSS_SVC_DUAL || param->solver_type == L2R_L1LOSS_SVC_DUAL)
	// 	alpha = Malloc(double, l);

	if(parent == -1)
	{
		for(int j=0; j<w_size; j++)
			w[j] = 0;
		// if(param->solver_type == L2R_L2LOSS_SVC_DUAL || param->solver_type == L2R_L1LOSS_SVC_DUAL)
		// {
		// 	for(int j=0; j<l; j++)
		// 		alpha[j] = 0;
		// }
	}
	else
	{
		for(int j=0; j<w_size; j++)
			w[j] = nodes[parent].w[j];
		// if(param->solver_type == L2R_L2LOSS_SVC_DUAL || param->solver_type == L2R_L1LOSS_SVC_DUAL)
		// 	for(int j=0; j<l; j++)
		// 		alpha[j] = nodes[parent].alpha[j];
	}

	// perceptron

	for(jj=0; jj < classCount[child]; jj++){
		int ind = labelInd[child][jj];
		//sub_prob_omp.y[ind] = +1;
		feature_node *x_tmp = prob->x[ind];
		double pred = sparse_operator::dot(w, x_tmp);
		if(pred < 0)
		{
			while(x_tmp->index != -1)
			{
				w[x_tmp->index-1] += x_tmp->value;
				x_tmp++;
			}
		}
	}


	train_one(&sub_prob_omp, param, w, alpha, weighted_C[child], param->C);

	// test distance
	if(parent != -1){
		double d1 = 0;
		for(int ll=0; ll<w_size; ll++)
			d1 += w[ll]*w[ll];

		printf("distance to 0: %4.5e\n", sqrt(d1));

		double d2 = 0;
		for(int ll=0; ll<w_size; ll++)
			d2 += (w[ll] - nodes[parent].w[ll])*(w[ll] - nodes[parent].w[ll]);

		printf("distance to parent: %4.5e\n", sqrt(d2));
	}

	printf("%ith label finished!\n", child+1);

	if(nodes[child+1].children.size() > 0)
	{
		for(int j=0; j<w_size; j++)
			nodes[child+1].w[j] = w[j];
		// if(param->solver_type == L2R_L2LOSS_SVC_DUAL || param->solver_type == L2R_L1LOSS_SVC_DUAL)
		// 	for(int j=0; j<l; j++)
		// 		nodes[child+1].alpha[j] = alpha[j];
	}

	int nzcount = 0;
	for(int j=0;j<w_size;j++){
		if(fabs(w[j]) < 0.0){
			w[j]=0;
		}
		else
		{
			nzcount++;
		}
	}
	//int start = totalnz;
	//totalnz += nzcount + 1;
	model_->w[child] = Malloc(feature_node, nzcount + 1);  // -1 for the last

	int cc = 0;
	int j;
	for(j=0;j<w_size;j++){
		if(w[j] != 0)
		{
			(model_->w[child]+cc)->index = j+1;
			(model_->w[child]+cc)->value = w[j];
			cc++;
		}
	}
	(model_->w[child]+cc)->index = -1;  // -1 for the last


	free(sub_prob_omp.y);
	free(w);
	free(alpha);


	// subproblem is solved now

	for(int i=0; i<nodes[node_idx].children.size(); i++)
	{
		int cc= nodes[node_idx].children[i];

		//dfs(model_, prob, param, nodes, cc);   // recursive call for dfs
		dfs(model_, prob, param, nodes, classCount, labelInd, weighted_C, cc, nr_class);

	}

	// free parent's w to save memory usage
	free(nodes[child+1].w);
	free(nodes[child+1].alpha);
	nodes[child+1].w = NULL;
	nodes[child+1].alpha = NULL;
}

int get_height(label_node *nodes, int node_idx)
{
	if(nodes[node_idx].children.size() == 0)
		return 0;
	int res = 0;
	for(int i=0; i<nodes[node_idx].children.size(); i++)
	{
		int cc = nodes[node_idx].children[i];
		int h = get_height(nodes, cc);
		res = max(res, h+1);
	}
	return res;
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	//int nr_class = 3786;
	int nr_class = 0;
	// get nr_class
	for(i=0;i<l;i++)
	{
		for(j=0; j<prob->numLabels[i]; j++)
		{
			nr_class = max(nr_class, (int)(*(prob->y[i]+j)) );
		}
	}
	printf("nr_class: %i\n", nr_class);


  // One v.s. classification
	int *label = NULL;
	int *start = NULL;
	int *count = NULL;
	//int *perm = Malloc(int,l);
	int **labelInd = Malloc(int *, nr_class);
	int *classCount = Malloc(int, nr_class);

	// group training data of the same class
	//group_classes(prob,&nr_class,&label,&start,&count,perm);
	label = Malloc(int, nr_class);

	for(i=0;i<nr_class;i++)
	{
		label[i] = i+1;
		classCount[i]=0;
	}


	for(i=0;i<l;i++)
	{
		int jj=0;
		while( jj < prob->numLabels[i] )  {
			int this_label = (int)(*(prob->y[i]+jj));
			//			printf("%d\n", this_label);
			classCount[this_label-1]++;
			jj++;
		}
	}

	for(i=0;i<nr_class;i++)
	{
	//			printf("class Count is class %d count = %d\n", i, classCount[i]);
		labelInd[i] = Malloc(int , classCount[i]);
	}


	int *label_ind_ind = Malloc(int, nr_class);
	for(i=0;i<nr_class;i++)
	{
		label_ind_ind[i] =0;
	}

	for(i=0;i<l;i++)
	{
		int jj=0;
		while( jj < prob->numLabels[i] )  {
			int this_label = (int)(*(prob->y[i]+jj));
			labelInd[this_label-1][label_ind_ind[this_label-1]] = i;
			label_ind_ind[this_label-1]++;
			jj++;
		}
	}


	model_->nr_class=nr_class;
	model_->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++)
		model_->label[i] = label[i];

	// calculate weighted C
	double *weighted_C = Malloc(double, nr_class);
	for(i=0;i<nr_class;i++)
		weighted_C[i] = param->C;
	for(i=0;i<param->nr_weight;i++)
	{
		for(j=0;j<nr_class;j++)
			if(param->weight_label[i] == label[j])
				break;
		if(j == nr_class)
			fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
		else
			weighted_C[j] *= param->weight[i];
	}

	// constructing the subproblem
	feature_node **x = Malloc(feature_node *,l);
	//for(i=0;i<l;i++)
	//	x[i] = prob->x[perm[i]];

	for(i=0;i<l;i++){
		x[i] = prob->x[i];
	}

	int k;
	subproblem sub_prob;
	sub_prob.l = l;
	sub_prob.n = n;
	sub_prob.x = Malloc(feature_node *,sub_prob.l);
	sub_prob.y = Malloc(double,sub_prob.l);

	for(k=0; k<sub_prob.l; k++)
		sub_prob.x[k] = x[k];

	//int startInd = 0; int endInd = nr_class;

	//model_->w=Malloc(double, w_size*nr_class);
	model_->w = Malloc(feature_node *,nr_class);
	model_->nr_class=nr_class;

	model_->label = Malloc(int,nr_class);
	for(i=0;i<nr_class;i++){
		model_->label[i] = label[i];
		//	info("labels are %u\n", model_->label[i]);
	}

	//double *w=Malloc(double, w_size);
	// all negative initialization
	//double *w0 = NULL;



	label_node* nodes = Malloc(label_node, nr_class+1);
	for(i=0;i<(nr_class+1);i++)
	{
		nodes[i].id = i;
		nodes[i].visited = false;
		nodes[i].alpha = NULL;
		//nodes[i].isparent = false;
		nodes[i].neighbours.resize(0);
		nodes[i].parent = -1;
		nodes[i].children.resize(0);
	}

	int start_node = 0;

	std::vector<std::pair<int, int> > order (0);

	if(param->init_strat == 2)
	{
		order_schedule(prob, param, nr_class, nodes);

		//(nodes, start_node, nr_class, order);

		printf("using MST scheduling!\n");
		// for(int j=0; j<order.size(); j++)
		// {
		// 	printf("%d th problem: from %d to %d \n", j+1, order[j].first, order[j].second);
		// }
	}
	else if(param->init_strat == 1)
	{
		//order.resize(0);
		for(int j=1; j<=nr_class; j++)
		{
			nodes[0].children.push_back(j);
			nodes[j].parent = 0;
		}

		//for(int j=0; j<nr_class; j++)
		//	order.push_back( std::make_pair(0, j+1) );

		printf("using all negative initialization!\n");
	}
	else if(param->init_strat == 3)
	{
		for(int j=1; j<=nr_class; j++)
		{
			nodes[0].children.push_back(j);
			nodes[j].parent = 0;
		}

	}
	else
	{
		for(int j=1; j<=nr_class; j++)
		{
			nodes[0].children.push_back(j);
			nodes[j].parent = -1;
		}
	}



	// calculate for w0, initial problem;
	nodes[0].w = Malloc(double, w_size);
	// if(param->solver_type == L2R_L2LOSS_SVC_DUAL || param->solver_type == L2R_L1LOSS_SVC_DUAL)
	// 	nodes[0].alpha = Malloc(double, l);

	subproblem sub_prob_omp;
	sub_prob_omp.l = l;
	sub_prob_omp.n = n;
	sub_prob_omp.x = x;
	sub_prob_omp.y = Malloc(double,l);

	// initialize w
	for(j=0;j<w_size;j++){
		nodes[0].w[j] = 0;
	}
	// initialize alpha
	// if(param->solver_type == L2R_L2LOSS_SVC_DUAL || param->solver_type == L2R_L1LOSS_SVC_DUAL)
	// 	for(j=0;j<l;j++)
	// 		nodes[0].alpha[j] = 0;


	for(k=0; k <sub_prob.l; k ++){
		sub_prob_omp.y[k] = -1;
	}

	train_one(&sub_prob_omp, param, nodes[0].w, nodes[0].alpha, weighted_C[i], param->C);

	// printf("##################\n");
	// printf("##################\n");
	// printf("##################\n");
	//
	// train_one(&sub_prob_omp, param, nodes[0].w, nodes[0].alpha, weighted_C[i], param->C);
	//
	// printf("##################\n");
	// printf("##################\n");
	// printf("##################\n");
	if(param->init_strat == 3)
	{
		for(j=0; j < w_size; j++)
			nodes[0].w[j] = 0;
		nodes[0].w[w_size-1] = -1.;
		printf("use bias -1 initialization\n");
	}

	nodes[0].visited = true;

	// MST parallel
	// divided into subtrees
	std::vector<int> subroots = nodes[0].children;
	// random shuffle subtrees
	std::random_shuffle( subroots.begin(), subroots.end() );

	printf("number of subtrees: %d\n", nodes[0].children.size());

	int height = get_height(nodes, 0);
	printf("height of the whole tree: %d\n", height);

	//calculate all other nodes
	//std::vector<std::pair<int,int > >::iterator it;

	omp_set_num_threads(param->n_threads);


	// // accuracy test
	// int kk = 1022;
	// subproblem sub_prob_omp2;
	// sub_prob_omp2.l = l;
	// sub_prob_omp2.n = n;
	// sub_prob_omp2.x = prob->x;
	// sub_prob_omp2.y = Malloc(double,l);
	//
	// for(int k=0; k <l; k ++){
	// 	sub_prob_omp2.y[k] = -1;
	// }
	//
	// int jj;
	// for(jj=0; jj < classCount[kk]; jj++){
	// 	int ind = labelInd[kk][jj];
	// 	sub_prob_omp2.y[ind] = +1;
	// }
	// double * alpha = NULL;
	// train_one(&sub_prob_omp2, param, nodes[0].w, alpha, weighted_C[kk], param->C);
	//
	// printf("accuracy test ends!");
	//
	// // accuracy test ends


	#pragma omp parallel for schedule(dynamic,1)
	for(int kk=0; kk<subroots.size(); kk++)
	{

		// solve different subtrees in different threads
		int node_idx = subroots[kk];
		dfs(model_, prob, param, nodes, classCount, labelInd, weighted_C, node_idx, nr_class);


	// 	int parent = order[kk].first;
	// 	int child = order[kk].second-1;
	//
	// 	subproblem sub_prob_omp;
	// 	sub_prob_omp.l = l;
	// 	sub_prob_omp.n = n;
	// 	sub_prob_omp.x = x;
	// 	sub_prob_omp.y = Malloc(double,l);
	//
	// 	for(k=0; k <sub_prob.l; k ++){
	// 		sub_prob_omp.y[k] = -1;
	// 	}
	//
	// 	int jj;
	// 	for(jj=0; jj < classCount[child]; jj++){
	// 		int ind = labelInd[child][jj];
	// 		sub_prob_omp.y[ind] = +1;
	// 	}
	//
	//
	// 	if(nodes[child+1].isparent)
	// 		nodes[child+1].w = Malloc(double, w_size);
	//
	// 	double *w=Malloc(double, w_size);
	//
	// 	if(parent == -1)
	// 	{
	// 		for(int j=0; j<w_size; j++)
	// 			w[j] = 0;
	// 	}
	// 	else
	// 	{
	// 		for(int j=0; j<w_size; j++)
	// 			w[j] = nodes[parent].w[j];
	// 	}
	//
	// 	train_one(&sub_prob_omp, param, w, weighted_C[child], param->C);
	//
	// 	printf("%ith label finished!\n", child+1);
	//
	//
	// 	if(nodes[child+1].isparent)
	// 	{
	// 		for(int j=0; j<w_size; j++)
	// 			nodes[child+1].w[j] = w[j];
	// 	}
	//
	// 	int nzcount = 0;
	// 	for(int j=0;j<w_size;j++){
	// 		if(fabs(w[j]) < 0.01){
	// 			w[j]=0;
	// 		}
	// 		else
	// 		{
	// 			nzcount++;
	// 		}
	// 	}
	// 	//int start = totalnz;
	// 	//totalnz += nzcount + 1;
	// 	model_->w[child] = Malloc(feature_node, nzcount + 1);  // -1 for the last
	//
	// 	int cc = 0;
	// 	int j;
	// 	for(j=0;j<w_size;j++){
	// 		if(w[j] != 0)
	// 		{
	// 			(model_->w[child]+cc)->index = j+1;
	// 			(model_->w[child]+cc)->value = w[j];
	// 			cc++;
	// 		}
	// 	}
	// 	(model_->w[child]+cc)->index = -1;  // -1 for the last
	//
	//
	// 	free(sub_prob_omp.y);
	// 	free(w);
	//
	}



	//long long totalnz = 0;

	// for(i=0;i<nr_class;i++)
	// {
	// 	subproblem sub_prob_omp;
	// 	sub_prob_omp.l = l;
	// 	sub_prob_omp.n = n;
	// 	sub_prob_omp.x = x;
	// 	sub_prob_omp.y = Malloc(double,l);
	//
	// 	// initialize w
	// 	double *w=Malloc(double, w_size);
	// 	for(int j=0;j<w_size;j++){
	// 		w[j] = 0;
	// 	}
	//
	// 	for(k=0; k <sub_prob.l; k ++){
	// 		sub_prob_omp.y[k] = -1;
	// 	}
	//
	// 	int jj;
	// 	for(jj=0; jj < classCount[i]; jj++){
	// 		int ind = labelInd[i][jj];
	// 		sub_prob_omp.y[ind] = +1;
	// 	}
	//
	// 	if(w0 != NULL)
	// 	{
	// 		//printf("using all negative initialization!\n");
	// 		for(j=0;j<w_size;j++)
	// 			w[j] = w0[j];
	// 	}
	//
	//
	// 	train_one(&sub_prob_omp, param, w, weighted_C[i], param->C);
	//
	// 	printf("%ith label finished!\n", i);
	//
	//
	// 	int nzcount = 0;
	// 	for(int j=0;j<w_size;j++){
	// 		if(fabs(w[j]) < 0.01){
	// 			w[j]=0;
	// 		}
	// 		else
	// 		{
	// 			nzcount++;
	// 		}
	// 	}
	// 	//int start = totalnz;
	// 	//totalnz += nzcount + 1;
	// 	model_->w[i] = Malloc(feature_node, nzcount + 1);  // -1 for the last
	//
	// 	int cc = 0;
	// 	int j;
	// 	for(j=0;j<w_size;j++){
	// 		if(w[j] != 0)
	// 		{
	// 			(model_->w[i]+cc)->index = j+1;
	// 			(model_->w[i]+cc)->value = w[j];
	// 			cc++;
	// 		}
	// 	}
	// 	(model_->w[i]+cc)->index = -1;  // -1 for the last
	//
	//
	// 	free(sub_prob_omp.y);
	// 	free(w);
	//
	//
	// }
			//free(w);


	free(x);
	free(label);
	free(start);
	free(count);
	//free(perm);
	free(sub_prob.x);
	free(sub_prob.y);
	free(weighted_C);

	return model_;
}

// save and load model

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_L2LOSS_SVC_GD",
	"", "", "", NULL
};

void save_w(FILE *fp, int nr_class, int n, const struct model *model_)
{
  std::vector< std::vector<std::pair<int, double> > > w (n); // w size: ( n x k )
  int i;
  //printf("n: %d\n", n);
  //printf("nr_class: %d\n", nr_class);
  for(i=0; i<nr_class; i++){
    struct feature_node *wp = model_->w[i];
    while(wp->index != -1)
    {
      //printf("wp->index: %d\n", wp->index);
      int idx = wp->index;
      double val = wp->value;
      w[idx - 1].push_back( std::make_pair(i+1, val) );   // index start from 1
      wp++;
    }
  }

  //printf("construct w complete\n");

  for(i=0; i<n; i++)
  {
    fprintf(fp, "%i ", w[i].size());
    for(int j=0; j<w[i].size(); j++)
    {
      fprintf(fp, "%d", w[i][j].first);
      fprintf(fp, ":");
      fprintf(fp, "%.4lf ", w[i][j].second);
    }
    fprintf(fp, "\n");
  }
  return;
}


int save_model(const char *model_file_name, const struct model *model_)
{
  //printf("start save model\n");
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	int nr_w;
	//if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
	//	nr_w=1;
	//else
	nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	//sparse output
  save_w(fp, nr_w, n, model_);
	// for(i=0; i<nr_class; i++)
	// {
	// 	struct feature_node *wp = model_->w[i];
	// 	while(wp->index != -1)
	// 	{
	// 		fprintf(fp, "%i", wp->index);
	// 		fprintf(fp, ":");
	// 		fprintf(fp, "%.16g ", wp->value);
	// 		wp++
	// 	}
	// 	fprintf(fp, "\n");
	// }
	// for(i=0; i<w_size; i++)
	// {
	// 	int j;
	// 	for(j=0; j<nr_w; j++)
	// 		fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
	// 	fprintf(fp, "\n");
	// }

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//
// FSCANF helps to handle fscanf failures.
// Its do-while block avoids the ambiguity when
// if (...)
//    FSCANF();
// is used
//
#define FSCANF(_stream, _format, _var)do\
{\
	if (fscanf(_stream, _format, _var) != 1)\
	{\
		fprintf(stderr, "ERROR: fscanf failed to read the model\n");\
		EXIT_LOAD_MODEL()\
	}\
}while(0)
// EXIT_LOAD_MODEL should NOT end with a semicolon.
#define EXIT_LOAD_MODEL()\
{\
	setlocale(LC_ALL, old_locale);\
	free(model_->label);\
	free(model_);\
	free(old_locale);\
	return NULL;\
}

struct model *load_model_stat(const char *model_file_name)
{
  FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	//int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				EXIT_LOAD_MODEL()
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

  setlocale(LC_ALL, old_locale);
  free(old_locale);

  if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

  return model_;
}

struct feature_node **load_w(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = setlocale(LC_ALL, NULL);
	if (old_locale)
	{
		old_locale = strdup(old_locale);
	}
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		FSCANF(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			FSCANF(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");
				EXIT_LOAD_MODEL()
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			FSCANF(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			FSCANF(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			FSCANF(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int nr_class = model_->nr_class;
			model_->label = Malloc(int,nr_class);
			for(int i=0;i<nr_class;i++)
				FSCANF(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			EXIT_LOAD_MODEL()
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;

	//int w_size = n;
	int nr_w;
	//if(nr_class==2 && param.solver_type != MCSVM_CS)
	//	nr_w = 1;
	//else
	nr_w = nr_class;
  //printf("n: %d\n", n);
  struct feature_node **W = Malloc(struct feature_node *, n);
	long long total_nz = 0;
  for(i=0; i<n; i++)
  {
    int j;
    int nnz;
    FSCANF(fp, "%d ", &nnz);
		total_nz += nnz;
    //printf("nnz: %d\n", nnz);
    W[i] = Malloc(struct feature_node, nnz+1);
    struct feature_node *wp = W[i];
    for(j=0; j<nnz; j++)
    {
      FSCANF(fp, "%d", &wp->index);
      if (fscanf(fp, ":") != 0)
      {
        fprintf(stderr, "ERROR: fscanf failed to w\n");
    	 	EXIT_LOAD_MODEL()
      }
      FSCANF(fp, "%lf ", &wp->value);
      wp++;
    }
    wp->index = -1;
    if (fscanf(fp, "\n") !=0)
    {
      fprintf(stderr, "ERROR: fscanf failed to read the model\n");
    	EXIT_LOAD_MODEL()
    }
  }
  //printf("test load model: W[0]->index: %d\n", (W[0]+1)->index);
	printf("total # nnz: %lf\n", total_nz);
	// model_->w=Malloc(double, w_size*nr_w);
	// for(i=0; i<w_size; i++)
	// {
	// 	int j;
	// 	for(j=0; j<nr_w; j++)
	// 		FSCANF(fp, "%lf ", &model_->w[i*nr_w+j]);
	// 	if (fscanf(fp, "\n") !=0)
	// 	{
	// 		fprintf(stderr, "ERROR: fscanf failed to read the model\n");
	// 		EXIT_LOAD_MODEL()
	// 	}
	// }

	setlocale(LC_ALL, old_locale);
	free(old_locale);
  free_and_destroy_model(&model_);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return W;
}


int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, long long* label)
{
	if (model_->label != NULL)
		for(long long i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}


// predict
int ** predict(struct feature_node **x, const model *model_, struct feature_node **W, int nr_test, int k, int n_threads)
{
  int nr_class = model_->nr_class;
  int n = model_->nr_feature;
  int ** res = Malloc(int *, nr_test);
  int i;
  //printf("start prediction, nr_test = %d\n", nr_test);
	//double *score = Malloc(double, nr_class);

	if(model_->bias >=0)
		n = n+1;

	omp_set_num_threads(n_threads);

	#pragma omp parallel for schedule(dynamic,5)
  for(i=0; i<nr_test; i++)
  {
    //printf("the %d th test sample\n", i);
    //struct feature_node *x_i = x[i];
		double score[nr_class];

    for(int j=0; j<nr_class; j++)
    {
      score[j] = 0;
    }
    res[i] = Malloc(int, k);
    struct feature_node *x_i = x[i];
    while(x_i->index != -1)
    {
      //printf("x[i]->index: %d\n", x_i->index);
			if(x_i->index > n)
				continue;
      struct feature_node *wp = W[ x_i->index - 1 ];
      while(wp->index != -1)
      {
        //printf("wp->index: %d\n", wp->index);
        score[ wp->index - 1 ] += wp->value * x_i->value;
        wp++;
      }
      x_i++;
    }
    //for(int j = 0; j <nr_class; j++)
    //  printf("score[%d]: %lf\n", j, score[j]);
    // find top k score
    std::priority_queue< std::pair<double, int>, std::vector< std::pair<double, int> >, std::greater <std::pair<double, int> > > q;
    for(int j=0; j<nr_class; j++)
    {
      if(q.size() < k)
        q.push(std::make_pair(score[j], j) );
      else if(q.top().first < score[j])
      {
        q.pop();
        q.push(std::make_pair(score[j], j) );
      }
    }

    int * topk_index = Malloc(int, k);
    for(int j=0; j<k; j++)
    {
      topk_index[k-j-1] = q.top().second;
      q.pop();
    }
    for(int j=0; j<k; j++)
    {
      res[i][j] =  model_->label[ topk_index[j] ] ;
      //printf("topk_index[j]: %d\n", topk_index[j]);
      //printf("res[i][j]: %d\n", res[i][j]);
    }
		int kk = i;
		if(kk%10000 == 0)
			printf("%d complete!\n", kk);
  }
	//free(score); // free score

  return res;
}

void evaluate(int ** pred, struct problem * test_prob, int k)
{
  int k1 = 1, k2 = 3, k3 =5;
  double p1 = 0, p3 =0, p5 = 0;
  double ndcg1 = 0, ndcg3 = 0, ndcg5 = 0;

  for(int i=0; i<test_prob->l; i++)
  {
    double deno1 = 0, deno3 = 0, deno5 = 0;
    double dcg1 = 0, dcg3 = 0, dcg5 = 0;
    int j=0;
    //int startidx = 0;

    for(; j<k1; j++)
    {
      for(int q=0; q<test_prob->numLabels[i]; q++)
      {
        if(pred[i][j] == (test_prob->y[i][q]) )
        {
          p1++;
          p3++;
          p5++;
          dcg1 += log(2)/log(j+2);
          dcg3 += log(2)/log(j+2);
          dcg5 += log(2)/log(j+2);
        }
      }
    }
    for(; j<k2; j++)
    {
      for(int q=0; q<test_prob->numLabels[i]; q++)
      {
        if(pred[i][j] == (test_prob->y[i][q]) )
        {
          p3++;
          p5++;
          dcg3 += log(2)/log(j+2);
          dcg5 += log(2)/log(j+2);
        }
      }
    }
    for(; j<k3; j++)
    {
      for(int q=0; q<test_prob->numLabels[i]; q++)
      {
        if(pred[i][j] == (test_prob->y[i][q]) )
        {
          p5++;
          dcg5 += log(2)/log(j+2);
        }
      }
    }

    for(j=0; j<min(k1, test_prob->numLabels[i]); j++)
    {
      deno1 += log(2)/log(j+2);
    }

    for(j=0; j<min(k2, test_prob->numLabels[i]); j++)
    {
      deno3 += log(2)/log(j+2);
    }

    for(j=0; j<min(k3, test_prob->numLabels[i]); j++)
    {
      deno5 += log(2)/log(j+2);
    }


    //startidx += test_prob->numLabels[i];
    ndcg1 += dcg1/deno1;
    ndcg3 += dcg3/deno3;
    ndcg5 += dcg5/deno5;

  }
  p1 = (p1*100.0)/(test_prob->l*k1);
  p3 = (p3*100.0)/(test_prob->l*k2);
  p5 = (p5*100.0)/(test_prob->l*k3);
  ndcg1 = (ndcg1*100.0)/test_prob->l;
  ndcg3 = (ndcg3*100.0)/test_prob->l;
  ndcg5 = (ndcg5*100.0)/test_prob->l;
  printf("precision at 1: %.5f \n precision at 3: %.5f \n precision at 5: %.5f \n", p1, p3, p5);
  printf("ndcg at 1: %.5f \n ndcg at 3: %.5f \n ndcg at 5: %.5f \n", ndcg1, ndcg3, ndcg5);

}


// free and destroy
void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
	if(param->init_sol != NULL)
		free(param->init_sol);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->p < 0)
		return "p < 0";

	if(param->solver_type != L2R_LR
		// && param->solver_type != L2R_L2LOSS_SVC_DUAL
		// && param->solver_type != L2R_L1LOSS_SVC_DUAL
		&& param->solver_type != L2R_L2LOSS_SVC
		&& param->solver_type != L1R_L2LOSS_SVC
		&& param->solver_type != L1R_LR
		&& param->solver_type != L2R_L2LOSS_SVC_GD)
		return "unknown solver type";

	if(param->init_sol != NULL
		&& param->solver_type != L2R_LR && param->solver_type != L2R_L2LOSS_SVC)
		return "Initial-solution specification supported only for solver L2R_LR and L2R_L2LOSS_SVC";

	return NULL;
}
