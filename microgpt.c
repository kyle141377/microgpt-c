#include "microgpt.h"
#include <stdarg.h>

/* ============ Value Implementation ============ */

Value *value_new(float data) {
    Value *v = (Value *)calloc(1, sizeof(Value));
    v->data = data;
    v->grad = 0.0f;
    v->n_children = 0;
    v->children = NULL;
    v->local_grads = NULL;
    v->op_type = 0;
    v->is_param = 0;
    v->visited_flag = 0;
    return v;
}

static void value_set_children(Value *v, int n, ...) {
    va_list args;
    va_start(args, n);
    v->n_children = n;
    v->children = (Value **)malloc(n * sizeof(Value *));
    v->local_grads = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        v->children[i] = va_arg(args, Value *);
        v->local_grads[i] = (float)va_arg(args, double);
    }
    va_end(args);
}

Value *value_add(Value *a, Value *b) {
    Value *r = value_new(a->data + b->data);
    value_set_children(r, 2, a, 1.0, b, 1.0);
    return r;
}

Value *value_mul(Value *a, Value *b) {
    Value *r = value_new(a->data * b->data);
    value_set_children(r, 2, a, (double)b->data, b, (double)a->data);
    return r;
}

Value *value_pow_scalar(Value *a, float exp) {
    Value *r = value_new(powf(a->data, exp));
    value_set_children(r, 1, a, (double)(exp * powf(a->data, exp - 1)));
    return r;
}

Value *value_log(Value *a) {
    Value *r = value_new(logf(a->data));
    value_set_children(r, 1, a, (double)(1.0f / a->data));
    return r;
}

Value *value_exp(Value *a) {
    float ed = expf(a->data);
    Value *r = value_new(ed);
    value_set_children(r, 1, a, (double)ed);
    return r;
}

Value *value_relu(Value *a) {
    Value *r = value_new(fmaxf(0.0f, a->data));
    value_set_children(r, 1, a, (double)(a->data > 0 ? 1.0f : 0.0f));
    return r;
}

Value *value_div_scalar(Value *a, float b) {
    Value *r = value_new(a->data / b);
    value_set_children(r, 1, a, (double)(1.0f / b));
    return r;
}

Value *value_div(Value *a, Value *b) {
    /* a / b = a * b^-1 */
    float b_inv = 1.0f / b->data;
    Value *r = value_new(a->data * b_inv);
    value_set_children(r, 2, a, (double)b_inv, b, (double)(-a->data * b_inv * b_inv));
    return r;
}

Value *value_neg(Value *a) {
    Value *r = value_new(-a->data);
    value_set_children(r, 1, a, -1.0);
    return r;
}

Value *value_add_scalar(Value *a, float b) {
    Value *r = value_new(a->data + b);
    value_set_children(r, 1, a, 1.0);
    return r;
}

void value_free(Value *v) {
    if (!v) return;
    if (v->children) free(v->children);
    if (v->local_grads) free(v->local_grads);
    free(v);
}

/* ============ Autograd Backward ============ */

/*
 * Iterative topological sort using heap-allocated stack.
 * Equivalent to recursive:
 *   def build_topo(v):
 *       if v not in visited: visited.add(v); for c in v._children: build_topo(c); topo.append(v)
 *
 * Uses post-order traversal: push children, then append node after all children processed.
 */
static int topo_n = 0;
#define MAX_TOPO_NODES 2000000

static void build_topo(Value *root, Value **topo, int gen) {
    /* Stack entries: node + child index */
    typedef struct { Value *v; int ci; } Frame;
    Frame *stack = (Frame *)malloc(256 * 1024 * sizeof(Frame));
    int sp = 0; /* stack pointer */
    int nodes_visited = 0;

    /* Use visited_flag == gen to mean "already seen this generation" */
    stack[sp].v = root;
    stack[sp].ci = 0;
    sp++;

    while (sp > 0) {
        Frame *f = &stack[sp - 1];
        Value *v = f->v;

        /* First time seeing this node: mark it */
        if (f->ci == 0) {
            v->visited_flag = gen;
            nodes_visited++;
        }

        /* Process next child */
        if (f->ci < v->n_children) {
            Value *child = v->children[f->ci];
            f->ci++;
            /* Push unvisited child */
            if (child->visited_flag != gen) {
                if (sp >= 256 * 1024) {
                    fprintf(stderr, "build_topo: stack overflow at %d\n", sp);
                    break;
                }
                stack[sp].v = child;
                stack[sp].ci = 0;
                sp++;
            }
            /* else: child already visited this gen, skip */
        } else {
            /* All children done, append to topo */
            if (topo_n < MAX_TOPO_NODES) {
                topo[topo_n++] = v;
            }
            sp--;
        }
    }
    free(stack);
}

void value_backward(Value *root) {
    static int gen = 1;
    gen++;

    /* Estimate graph size: for 1 layer, 16 embed, 7 seq, vocab 23:
     * Each position ~50K nodes. 7 positions ~350K. Be generous. */
    int max_nodes = MAX_TOPO_NODES;
    Value **topo = (Value **)malloc(max_nodes * sizeof(Value *));
    if (!topo) { fprintf(stderr, "backward: alloc failed\n"); return; }

    topo_n = 0;
    build_topo(root, topo, gen);

    root->grad = 1.0f;
    for (int i = topo_n - 1; i >= 0; i--) {
        Value *v = topo[i];
        for (int j = 0; j < v->n_children; j++)
            v->children[j]->grad += v->local_grads[j] * v->grad;
    }
    free(topo);
}

/* Free the entire computation graph rooted at `root`.
 * Uses BFS to collect all reachable non-param nodes,
 * then frees them in reverse order (leaves first). */
static int free_gen = 100;
static void free_graph(Value *root) {
    free_gen++;
    int max_nodes = MAX_TOPO_NODES;
    Value **nodes = (Value **)malloc(max_nodes * sizeof(Value *));
    int count = 0;

    /* BFS to collect all non-param nodes */
    Value **queue = (Value **)malloc(max_nodes * sizeof(Value *));
    int head = 0, tail = 0;
    queue[tail++] = root;

    while (head < tail && count < max_nodes) {
        Value *v = queue[head++];
        if (v->is_param) continue;
        if (v->visited_flag == free_gen) continue; /* already collected */
        v->visited_flag = free_gen;

        nodes[count++] = v;
        for (int i = 0; i < v->n_children && tail < max_nodes; i++) {
            Value *child = v->children[i];
            if (child && !child->is_param)
                queue[tail++] = child;
        }
    }
    free(queue);

    /* Free in reverse order (leaves first) */
    for (int i = count - 1; i >= 0; i--) {
        nodes[i]->children = NULL;
        nodes[i]->local_grads = NULL;
        nodes[i]->n_children = 0;
        value_free(nodes[i]);
    }
    free(nodes);
}

/* ============ Matrix Implementation ============ */

static float rand_gauss(float std) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    while (u1 < 1e-10f) u1 = (float)rand() / (float)RAND_MAX;
    return std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

Matrix *matrix_new(int nout, int nin, float std) {
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    m->nout = nout;
    m->nin = nin;
    m->data = (Value ***)malloc(nout * sizeof(Value **));
    for (int i = 0; i < nout; i++) {
        m->data[i] = (Value **)malloc(nin * sizeof(Value *));
        for (int j = 0; j < nin; j++) {
            m->data[i][j] = value_new(rand_gauss(std));
            m->data[i][j]->is_param = 1;
        }
    }
    return m;
}

void matrix_free_values(Matrix *m) {
    if (!m || !m->data) return;
    for (int i = 0; i < m->nout; i++) {
        if (m->data[i]) {
            for (int j = 0; j < m->nin; j++) value_free(m->data[i][j]);
            free(m->data[i]);
        }
    }
    free(m->data);
    m->data = NULL;
}

/* ============ Model ============ */

StateDict *model_init(int vocab_size) {
    StateDict *sd = (StateDict *)calloc(1, sizeof(StateDict));
    float std = 0.08f;
    sd->wte = *matrix_new(vocab_size, N_EMBD, std);
    sd->wpe = *matrix_new(BLOCK_SIZE, N_EMBD, std);
    sd->lm_head = *matrix_new(vocab_size, N_EMBD, std);
    for (int i = 0; i < N_LAYER; i++) {
        sd->attn_wq[i] = *matrix_new(N_EMBD, N_EMBD, std);
        sd->attn_wk[i] = *matrix_new(N_EMBD, N_EMBD, std);
        sd->attn_wv[i] = *matrix_new(N_EMBD, N_EMBD, std);
        sd->attn_wo[i] = *matrix_new(N_EMBD, N_EMBD, std);
        sd->mlp_fc1[i] = *matrix_new(4 * N_EMBD, N_EMBD, std);
        sd->mlp_fc2[i] = *matrix_new(N_EMBD, 4 * N_EMBD, std);
    }
    return sd;
}

void model_free(StateDict *sd) {
    if (!sd) return;
    matrix_free_values(&sd->wte);
    matrix_free_values(&sd->wpe);
    matrix_free_values(&sd->lm_head);
    for (int i = 0; i < N_LAYER; i++) {
        matrix_free_values(&sd->attn_wq[i]);
        matrix_free_values(&sd->attn_wk[i]);
        matrix_free_values(&sd->attn_wv[i]);
        matrix_free_values(&sd->attn_wo[i]);
        matrix_free_values(&sd->mlp_fc1[i]);
        matrix_free_values(&sd->mlp_fc2[i]);
    }
    free(sd);
}

int model_count_params(StateDict *sd, int vocab_size) {
    int c = vocab_size * N_EMBD + BLOCK_SIZE * N_EMBD + vocab_size * N_EMBD;
    for (int i = 0; i < N_LAYER; i++)
        c += N_EMBD * N_EMBD * 4 + 4 * N_EMBD * N_EMBD * 2;
    return c;
}

/* ============ Forward Operations ============ */

static Value **linear(Value **x, int x_len, Matrix *w) {
    Value **out = (Value **)malloc(w->nout * sizeof(Value *));
    for (int i = 0; i < w->nout; i++) {
        Value *sum = value_new(0.0f);
        for (int j = 0; j < w->nin; j++) {
            Value *prod = value_mul(w->data[i][j], x[j]);
            Value *ns = value_add(sum, prod);
            sum = ns;
            /* Don't free sum or prod - they're part of the graph */
        }
        out[i] = sum;
    }
    return out;
}

static Value **rmsnorm(Value **x, int n) {
    Value *ms = value_new(0.0f);
    for (int i = 0; i < n; i++) {
        Value *sq = value_mul(x[i], x[i]);
        Value *nms = value_add(ms, sq);
        ms = nms;
        /* Don't free ms or sq */
    }
    ms = value_div_scalar(ms, (float)n);
    Value *ep = value_add_scalar(ms, 1e-5f);
    Value *scale = value_pow_scalar(ep, -0.5f);

    Value **out = (Value **)malloc(n * sizeof(Value *));
    for (int i = 0; i < n; i++) out[i] = value_mul(x[i], scale);
    return out;
}

static Value **softmax_arr(Value **logits, int n) {
    float mx = -FLT_MAX;
    for (int i = 0; i < n; i++) if (logits[i]->data > mx) mx = logits[i]->data;

    Value **exps = (Value **)malloc(n * sizeof(Value *));
    Value *total = value_new(0.0f);
    for (int i = 0; i < n; i++) {
        Value *d = value_add_scalar(logits[i], -mx);
        exps[i] = value_exp(d);
        Value *nt = value_add(total, exps[i]);
        total = nt;
        /* Don't free d, total, or exps[i] yet */
    }

    Value **probs = (Value **)malloc(n * sizeof(Value *));
    for (int i = 0; i < n; i++) {
        probs[i] = value_div(exps[i], total);
    }
    free(exps);
    return probs;
}

/* ============ GPT Forward (with autograd) ============ */

static Value **gpt_forward(StateDict *sd, int token_id, int pos_id,
                           Value ****kc, Value ****vc, int *cl, int vs) {
    Value *te[N_EMBD], *pe[N_EMBD];
    for (int i = 0; i < N_EMBD; i++) {
        te[i] = sd->wte.data[token_id][i];
        pe[i] = sd->wpe.data[pos_id][i];
    }
    Value *x[N_EMBD];
    for (int i = 0; i < N_EMBD; i++) x[i] = value_add(te[i], pe[i]);

    Value **xn = rmsnorm(x, N_EMBD);
    for (int i = 0; i < N_EMBD; i++) x[i] = xn[i];
    free(xn);

    for (int li = 0; li < N_LAYER; li++) {
        Value *xr[N_EMBD];
        for (int i = 0; i < N_EMBD; i++) xr[i] = x[i];

        xn = rmsnorm(x, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) x[i] = xn[i];
        free(xn);

        Value **q = linear(x, N_EMBD, &sd->attn_wq[li]);
        Value **k = linear(x, N_EMBD, &sd->attn_wk[li]);
        Value **v = linear(x, N_EMBD, &sd->attn_wv[li]);

        int t = cl[li];
        kc[li][t] = k;
        vc[li][t] = v;
        cl[li]++;
        int sl = cl[li];

        Value **xa = (Value **)malloc(N_EMBD * sizeof(Value *));
        for (int i = 0; i < N_EMBD; i++) xa[i] = value_new(0.0f);

        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;
            Value **al = (Value **)malloc(sl * sizeof(Value *));
            for (int t2 = 0; t2 < sl; t2++) {
                Value *sc = value_new(0.0f);
                for (int j = 0; j < HEAD_DIM; j++) {
                    Value *p = value_mul(q[hs+j], kc[li][t2][hs+j]);
                    Value *ns = value_add(sc, p);
                    sc = ns;
                }
                al[t2] = value_div_scalar(sc, sqrtf((float)HEAD_DIM));
            }
            Value **aw = softmax_arr(al, sl);
            free(al);

            for (int j = 0; j < HEAD_DIM; j++) {
                Value *w = value_new(0.0f);
                for (int t2 = 0; t2 < sl; t2++) {
                    Value *wv = value_mul(aw[t2], vc[li][t2][hs+j]);
                    Value *nw = value_add(w, wv);
                    w = nw;
                }
                xa[hs+j] = w;
            }
            free(aw);
        }

        Value **xo = linear(xa, N_EMBD, &sd->attn_wo[li]);
        for (int i = 0; i < N_EMBD; i++) {
            x[i] = value_add(xo[i], xr[i]);
        }
        /* Don't free x[i], xo[i], xr[i], q, k, v, xa - all part of graph */
        free(xo);
        free(q); free(k); free(v); free(xa);

        /* MLP */
        for (int i = 0; i < N_EMBD; i++) xr[i] = x[i];
        xn = rmsnorm(x, N_EMBD);
        for (int i = 0; i < N_EMBD; i++) x[i] = xn[i];
        free(xn);

        Value **f1 = linear(x, N_EMBD, &sd->mlp_fc1[li]);
        for (int i = 0; i < 4*N_EMBD; i++) { Value *r = value_relu(f1[i]); f1[i] = r; }
        Value **f2 = linear(f1, 4*N_EMBD, &sd->mlp_fc2[li]);

        for (int i = 0; i < N_EMBD; i++) {
            x[i] = value_add(f2[i], xr[i]);
        }
        free(f1); free(f2);
    }

    return linear(x, N_EMBD, &sd->lm_head);
}

/* ============ Collect params for optimizer ============ */

static void get_all_params(StateDict *sd, Value ***params, int *idx, int vs) {
    #define GP(m) do { for(int i=0;i<(m).nout;i++) for(int j=0;j<(m).nin;j++) (*params)[(*idx)++]=(m).data[i][j]; } while(0)
    GP(sd->wte); GP(sd->wpe);
    for(int l=0;l<N_LAYER;l++) {
        GP(sd->attn_wq[l]); GP(sd->attn_wk[l]); GP(sd->attn_wv[l]); GP(sd->attn_wo[l]);
        GP(sd->mlp_fc1[l]); GP(sd->mlp_fc2[l]);
    }
    GP(sd->lm_head);
    #undef GP
}

/* ============ Dataset ============ */

Dataset *dataset_load(const char *path) {
    Dataset *d = (Dataset *)calloc(1, sizeof(Dataset));
    d->docs = (char **)malloc(MAX_DOCS * sizeof(char *));
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", path); exit(1); }
    char line[MAX_DOC_LEN];
    while (fgets(line, sizeof(line), f)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len]='\0';
        if (len==0) continue;
        d->docs[d->n_docs++] = strdup(line);
        if (d->n_docs >= MAX_DOCS) break;
    }
    fclose(f);

    int seen[256]={0}, uc=0;
    for (int di=0;di<d->n_docs;di++)
        for (int ci=0;d->docs[di][ci];ci++) {
            unsigned char c=(unsigned char)d->docs[di][ci];
            if (!seen[c]) { seen[c]=1; d->uchars[uc++]=(char)c; }
        }
    for (int i=0;i<uc-1;i++) for(int j=i+1;j<uc;j++)
        if (d->uchars[i]>d->uchars[j]) { char t=d->uchars[i]; d->uchars[i]=d->uchars[j]; d->uchars[j]=t; }
    d->vocab_size = uc + 1;
    d->bos_token = uc;
    printf("num docs: %d\nvocab size: %d\n", d->n_docs, d->vocab_size);
    return d;
}

void dataset_free(Dataset *d) {
    if (!d) return;
    for (int i=0;i<d->n_docs;i++) free(d->docs[i]);
    free(d->docs); free(d);
}

/* ============ Inference (pure float, no autograd) ============ */

static int sample_wt(int n, float *w) {
    float r=(float)rand()/RAND_MAX, c=0.0f;
    for(int i=0;i<n;i++) { c+=w[i]; if(r<c) return i; }
    return n-1;
}

void inference(StateDict *sd, Dataset *ds, int ns, float temp) {
    printf("\n--- inference (new, hallucinated names) ---\n");
    for (int si=0;si<ns;si++) {
        int tid = ds->bos_token;
        char samp[MAX_DOC_LEN];
        int sl = 0;
        float kc[N_LAYER][BLOCK_SIZE][N_EMBD], vc[N_LAYER][BLOCK_SIZE][N_EMBD];
        int cl[N_LAYER] = {0};

        for (int pos=0;pos<BLOCK_SIZE;pos++) {
            float x[N_EMBD];
            for(int i=0;i<N_EMBD;i++) x[i]=sd->wte.data[tid][i]->data+sd->wpe.data[pos][i]->data;
            float ms=0;
            for(int i=0;i<N_EMBD;i++) ms+=x[i]*x[i];
            ms/=N_EMBD;
            float sc=1.0f/sqrtf(ms+1e-5f);
            for(int i=0;i<N_EMBD;i++) x[i]*=sc;

            for(int li=0;li<N_LAYER;li++) {
                float xr[N_EMBD]; memcpy(xr,x,sizeof(x));
                ms=0; for(int i=0;i<N_EMBD;i++) ms+=x[i]*x[i];
                ms/=N_EMBD; sc=1.0f/sqrtf(ms+1e-5f); for(int i=0;i<N_EMBD;i++) x[i]*=sc;

                float q[N_EMBD],k[N_EMBD],v[N_EMBD];
                memset(q,0,sizeof(q));memset(k,0,sizeof(k));memset(v,0,sizeof(v));
                for(int i=0;i<N_EMBD;i++) for(int j=0;j<N_EMBD;j++) {
                    q[i]+=sd->attn_wq[li].data[i][j]->data*x[j];
                    k[i]+=sd->attn_wk[li].data[i][j]->data*x[j];
                    v[i]+=sd->attn_wv[li].data[i][j]->data*x[j];
                }

                int t=cl[li];
                memcpy(kc[li][t],k,sizeof(k));memcpy(vc[li][t],v,sizeof(v));cl[li]++;
                int seq=cl[li];

                float xa[N_EMBD]={0};
                for(int h=0;h<N_HEAD;h++) {
                    int hs=h*HEAD_DIM;
                    float la[BLOCK_SIZE];
                    for(int t2=0;t2<seq;t2++) {
                        float s=0; for(int j=0;j<HEAD_DIM;j++) s+=q[hs+j]*kc[li][t2][hs+j];
                        la[t2]=s/sqrtf((float)HEAD_DIM);
                    }
                    float mx=-FLT_MAX,es=0,aw[BLOCK_SIZE];
                    for(int t2=0;t2<seq;t2++) if(la[t2]>mx) mx=la[t2];
                    for(int t2=0;t2<seq;t2++) { aw[t2]=expf(la[t2]-mx); es+=aw[t2]; }
                    for(int t2=0;t2<seq;t2++) aw[t2]/=es;
                    for(int j=0;j<HEAD_DIM;j++) for(int t2=0;t2<seq;t2++) xa[hs+j]+=aw[t2]*vc[li][t2][hs+j];
                }

                float xo[N_EMBD]={0};
                for(int i=0;i<N_EMBD;i++) { for(int j=0;j<N_EMBD;j++) xo[i]+=sd->attn_wo[li].data[i][j]->data*xa[j]; x[i]=xo[i]+xr[i]; }

                memcpy(xr,x,sizeof(x));
                ms=0; for(int i=0;i<N_EMBD;i++) ms+=x[i]*x[i];
                ms/=N_EMBD; sc=1.0f/sqrtf(ms+1e-5f); for(int i=0;i<N_EMBD;i++) x[i]*=sc;

                float f1[4*N_EMBD];
                for(int i=0;i<4*N_EMBD;i++) { f1[i]=0; for(int j=0;j<N_EMBD;j++) f1[i]+=sd->mlp_fc1[li].data[i][j]->data*x[j]; if(f1[i]<0) f1[i]=0; }
                float f2[N_EMBD]={0};
                for(int i=0;i<N_EMBD;i++) { for(int j=0;j<4*N_EMBD;j++) f2[i]+=sd->mlp_fc2[li].data[i][j]->data*f1[j]; x[i]=f2[i]+xr[i]; }
            }

            int vs=ds->vocab_size;
            float lo[VOCAB_MAX]={0};
            for(int i=0;i<vs;i++) for(int j=0;j<N_EMBD;j++) lo[i]+=sd->lm_head.data[i][j]->data*x[j];

            float mx=-FLT_MAX,es=0,pr[VOCAB_MAX];
            for(int i=0;i<vs;i++) { float t=lo[i]/temp; if(t>mx) mx=t; }
            for(int i=0;i<vs;i++) { pr[i]=expf(lo[i]/temp-mx); es+=pr[i]; }
            for(int i=0;i<vs;i++) pr[i]/=es;

            tid = sample_wt(vs, pr);
            if (tid==ds->bos_token) break;
            samp[sl++] = ds->uchars[tid];
        }
        samp[sl]='\0';
        printf("sample %2d: %s\n", si+1, samp);
    }
}

static void shuffle_docs(Dataset *d) {
    for(int i=d->n_docs-1;i>0;i--) {
        int j=rand()%(i+1);
        char *t=d->docs[i]; d->docs[i]=d->docs[j]; d->docs[j]=t;
    }
}

/* ============ Training with autograd ============ */

static void train_autograd(StateDict *sd, Dataset *ds, int steps) {
    int vs = ds->vocab_size;
    int np = model_count_params(sd, vs);
    Value **params = (Value **)malloc(np * sizeof(Value *));
    int pidx = 0;
    get_all_params(sd, &params, &pidx, vs);

    float *adam_m = (float *)calloc(np, sizeof(float));
    float *adam_v = (float *)calloc(np, sizeof(float));

    /* KV cache */
    Value ****kc = (Value ****)malloc(N_LAYER * sizeof(Value ***));
    Value ****vc = (Value ****)malloc(N_LAYER * sizeof(Value ***));
    for(int l=0;l<N_LAYER;l++) {
        kc[l] = (Value ***)malloc(BLOCK_SIZE * sizeof(Value **));
        vc[l] = (Value ***)malloc(BLOCK_SIZE * sizeof(Value **));
    }
    int cl[N_LAYER];

    const float lr=0.01f, b1=0.85f, b2=0.99f, eps=1e-8f;

    printf("Training %d params for %d steps (autograd)...\n", np, steps);

    for (int step=0; step<steps; step++) {
        for(int l=0;l<N_LAYER;l++) cl[l]=0;
        const char *doc = ds->docs[step % ds->n_docs];

        int tok[MAX_DOC_LEN];
        int nt=1; tok[0]=ds->bos_token;
        for(int i=0;doc[i];i++)
            for(int c=0;c<vs-1;c++) if(ds->uchars[c]==doc[i]) { tok[nt++]=c; break; }
        tok[nt++]=ds->bos_token;
        int n = (nt-1<BLOCK_SIZE) ? (nt-1) : BLOCK_SIZE;

        /* Forward */
        Value **all_logits = (Value **)malloc(n * vs * sizeof(Value *));
        Value **all_probs = (Value **)malloc(n * vs * sizeof(Value *));
        Value **losses = (Value **)malloc(n * sizeof(Value *));
        for(int pos=0;pos<n;pos++) {
            int ti=tok[pos], tai=tok[pos+1];
            Value **logits = gpt_forward(sd, ti, pos, kc, vc, cl, vs);
            Value **probs = softmax_arr(logits, vs);
            losses[pos] = value_neg(value_log(probs[tai]));
            /* Store for later cleanup */
            for(int i=0;i<vs;i++) {
                all_logits[pos*vs+i] = logits[i];
                all_probs[pos*vs+i] = probs[i];
            }
            free(logits); free(probs);
        }

        Value *ls = value_new(0.0f);
        for(int i=0;i<n;i++) {
            Value *ns = value_add(ls, losses[i]);
            ls = ns;
        }
        Value *loss = value_div_scalar(ls, (float)n);

        /* Backward */
        value_backward(loss);

        /* Adam update */
        float lr_t = lr * (1.0f - (float)step/steps);
        for(int i=0;i<np;i++) {
            adam_m[i] = b1*adam_m[i] + (1-b1)*params[i]->grad;
            adam_v[i] = b2*adam_v[i] + (1-b2)*params[i]->grad*params[i]->grad;
            float mh = adam_m[i] / (1-powf(b1, step+1));
            float vh = adam_v[i] / (1-powf(b2, step+1));
            params[i]->data -= lr_t * mh / (sqrtf(vh)+eps);
            params[i]->grad = 0;
        }

        printf("step %4d / %4d | loss %.4f\n", step+1, steps, loss->data);
        fflush(stdout);

        /* Free the entire computation graph */
        free_graph(loss);
        free(losses);
        free(all_logits);
        free(all_probs);
        /* Clear KV cache pointers (nodes already freed) */
        for(int l=0;l<N_LAYER;l++) {
            for(int t=0;t<cl[l];t++) {
                kc[l][t] = NULL;
                vc[l][t] = NULL;
            }
        }
    }
    printf("\n");

    free(params); free(adam_m); free(adam_v);
    for(int l=0;l<N_LAYER;l++) { free(kc[l]); free(vc[l]); }
    free(kc); free(vc);
}

/* ============ Main ============ */

int main(int argc, char *argv[]) {
    srand(42);

    const char *path = "input.txt";
    if (argc > 1) path = argv[1];

    FILE *t = fopen(path, "r");
    if (!t) { fprintf(stderr, "File %s not found.\n", path); return 1; }
    fclose(t);

    Dataset *ds = dataset_load(path);
    shuffle_docs(ds);

    StateDict *sd = model_init(ds->vocab_size);
    printf("num params: %d\n", model_count_params(sd, ds->vocab_size));

    train_autograd(sd, ds, 1000);

    inference(sd, ds, 20, 0.5f);

    dataset_free(ds);
    model_free(sd);
    printf("\nDone!\n");
    return 0;
}
