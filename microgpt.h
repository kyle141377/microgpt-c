#ifndef MICROGPT_H
#define MICROGPT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define N_LAYER     1
#define N_EMBD      16
#define BLOCK_SIZE  16
#define N_HEAD      4
#define HEAD_DIM    (N_EMBD / N_HEAD)
#define VOCAB_MAX   256
#define NUM_STEPS   1000
#define MAX_DOC_LEN 256
#define MAX_DOCS    10000

#define LR          0.01f
#define BETA1       0.85f
#define BETA2       0.99f
#define EPS_ADAM    1e-8f
#define INF_TEMP    0.5f

typedef struct Value {
    float data;
    float grad;
    int n_children;
    struct Value **children;
    float *local_grads;
    int op_type;
    int is_param;
    int visited_flag;
} Value;

typedef struct {
    Value ***data;
    int nout, nin;
} Matrix;

typedef struct {
    Matrix wte;
    Matrix wpe;
    Matrix attn_wq[N_LAYER];
    Matrix attn_wk[N_LAYER];
    Matrix attn_wv[N_LAYER];
    Matrix attn_wo[N_LAYER];
    Matrix mlp_fc1[N_LAYER];
    Matrix mlp_fc2[N_LAYER];
    Matrix lm_head;
} StateDict;

typedef struct {
    float *m;
    float *v;
    int n_params;
} AdamState;

typedef struct {
    char **docs;
    int n_docs;
    int vocab_size;
    char uchars[VOCAB_MAX];
    int bos_token;
} Dataset;

Value *value_new(float data);
Value *value_add(Value *a, Value *b);
Value *value_mul(Value *a, Value *b);
Value *value_pow_scalar(Value *a, float exp);
Value *value_log(Value *a);
Value *value_exp(Value *a);
Value *value_relu(Value *a);
Value *value_div_scalar(Value *a, float b);
Value *value_neg(Value *a);
Value *value_add_scalar(Value *a, float b);
void value_free(Value *v);

Matrix *matrix_new(int nout, int nin, float std);
void matrix_free_values(Matrix *m);

StateDict *model_init(int vocab_size);
void model_free(StateDict *sd);
int model_count_params(StateDict *sd, int vocab_size);

Dataset *dataset_load(const char *path);
void dataset_free(Dataset *d);

void inference(StateDict *sd, Dataset *ds, int n_samples, float temperature);

#endif
