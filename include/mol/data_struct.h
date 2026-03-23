#ifndef _MOL_DATA_STRUCT_H_
#define _MOL_DATA_STRUCT_H_
#include <stdlib.h>
#include <math.h>
typedef struct vector_node_struct
{
    int data;
    struct vector_node_struct *next;
} vector_node;

typedef struct
{
    vector_node *head;
    int len;
} qvector;

typedef struct vvector_node_struct
{
    qvector data;
    struct vvector_node_struct *next;
} vvector_node;

typedef struct
{
    vvector_node *head;
    int len;
} qvvector;
void init_vector(qvector *vector);
void push_vector(qvector *vector, int data);
void pop_vector(qvector *vector);
void del_vector(qvector *vector);
int get_vector_data(qvector *vector, int i);
void init_vvecotr(qvvector *vvector);
void push_vvecotr(qvvector *vvecotr);
void pop_vvector(qvvector *vvector);
void del_vvector(qvvector *vvector);
qvector *get_vvector_data(qvvector *vvector, int i);

typedef struct tree_node_struct
{
    int data;
    int key;
    struct tree_node_struct *left_leaf;
    struct tree_node_struct *right_leaf;
} tree_node;

typedef struct
{
    int node_num;
    tree_node *node_list;
    tree_node *head;
} qtree;

void init_tree(qtree *tree);
void make_tree_node_son(tree_node *tree_l, int *list, int *klist, int len, tree_node **node,
                        int *slist_1, int *slist_2, int *skey_1, int *skey_2, int *slen_1, int *slen_2, int *nn, int *);
void make_tree(int *list, int *klist, int len, qtree *tree);
int find_tree_node(int data, tree_node *node);
int find_tree(int data, qtree *tree);
void del_tree_node(qtree *tree);
void del_tree(qtree *tree);

typedef struct
{
    int i, j;
} binary_num;

typedef struct
{
    binary_num *data;
    int len;
} binary_list;
void sort_binary_list(binary_list *bl);
#endif
