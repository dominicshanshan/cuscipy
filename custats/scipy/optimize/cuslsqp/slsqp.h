typedef struct {
    double t, f0, h1, h2, h3, h4;
    int n1, n2, n3;
    double t0, gs;
    double tol;
    int line;
    double alpha;
    int iexact;
    int incons, ireset, itermx;
    double *x0;
} slsqpb_state;

int slsqp(int *m, int *meq, int *la, int *n, double *x, const double *xl, const double *xu, double *f, const double *c__, const double *g, const double *a, double *acc, int *iter, int *mode, double *w, int *l_w__, int * jw, int *l_jw__, slsqpb_state *state);