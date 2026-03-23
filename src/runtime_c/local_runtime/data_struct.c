#include "mol/data_struct.h"
void init_vector(qvector* vector)
{
    vector->head=NULL;
    vector->len=0;
    return;
}

void push_vector(qvector* vector, int data)
{
    if (vector->len==0)
    {
        vector->head=(vector_node*)malloc(sizeof(vector_node));
        vector->head->data=data;
        vector->head->next=NULL;
        vector->len++;
        return;
    }
    vector_node* new_node=(vector_node*)malloc(sizeof(vector_node));
    new_node->data=data;
    new_node->next=vector->head;
    vector->head=new_node;
    vector->len++;
    return;
}

void pop_vector(qvector* vector)
{
    vector_node* tmp;
    tmp=vector->head;
    vector->head=vector->head->next;
    free(tmp);
    vector->len--;
    return;
}

void del_vector(qvector* vector)
{
    do
    {
        if (vector->len==0) break;
        pop_vector(vector);
    } while (1);
    vector->head=NULL;
    return;
}

int get_vector_data(qvector* vector, int i)
{
    vector_node* node=vector->head;
    for (int j=0; j<i; j++)
    {
        node=node->next;
    }
    return node->data;
}

void init_vvecotr(qvvector* vvector)
{
    vvector->len=0;
    vvector->head=NULL;
}

void push_vvecotr(qvvector* vvector)
{
    if (vvector->len==0)
    {
        vvector->head=(vvector_node*)malloc(sizeof(vvector_node));
        init_vector(&vvector->head->data);
        vvector->head->next=NULL;
        vvector->len++;
        return;
    }
    vvector_node* new_node=(vvector_node*)malloc(sizeof(vvector_node));
    init_vector(&new_node->data);
    new_node->next=vvector->head;
    vvector->head=new_node;
    vvector->len++;
    return;
}
void pop_vvector(qvvector* vvector)
{
    vvector_node* tmp;
    tmp=vvector->head;
    vvector->head=vvector->head->next;
    del_vector(&tmp->data);
    free(tmp);
    vvector->len--;
    return;
}
void del_vvector(qvvector* vvector)
{ 
    do
    {
        if (vvector->len==0) break;
        pop_vvector(vvector);
    } while (1);
    vvector->head=NULL;
    return;
}
qvector* get_vvector_data(qvvector* vvector, int i)
{
    vvector_node* node=vvector->head;
    for (int j=0; j<i; j++)
    {
        node=node->next;
    }
    return &(node->data);
}

void init_tree(qtree* tree)
{
    tree->head=NULL;
    tree->node_num=0;
    return;
}

void make_tree_node_son(tree_node* tree_l, int* list, int* klist, int len, tree_node** node, 
int* slist_1, int* slist_2, int* skey_1, int* skey_2,int* slen_1, int* slen_2, int* nn, int* head_p)
{
    int* slist_11=NULL,* slist_12=NULL;
    int* skey_11=NULL,* skey_12=NULL;
    int slen_11, slen_12;
    int* slist_21=NULL,* slist_22=NULL;
    int* skey_21=NULL,* skey_22=NULL;
    int slen_21, slen_22;
    int data_p=ceil(len*0.5);
    *node=&tree_l[*nn];
    (*node)->data=list[data_p-1];
    (*node)->key=klist[data_p-1];
    (*node)->left_leaf=NULL;
    (*node)->right_leaf=NULL;
    *slen_1=data_p-1;
    *slen_2=len-data_p;
    if (head_p!=NULL) *head_p=*nn;
    if (*slen_1>0)
    {
        (*nn)++;
        slist_1=list;
        skey_1=klist;
        make_tree_node_son(tree_l,slist_1,skey_1,*slen_1,&((*node)->left_leaf),
        slist_11,slist_12,skey_11,skey_12,&slen_11,&slen_12,nn,NULL);
    }
    if (*slen_2>0)
    {
        (*nn)++;
        slist_2=&list[data_p];
        skey_2=&klist[data_p];
        make_tree_node_son(tree_l,slist_2,skey_2,*slen_2,&((*node)->right_leaf),
        slist_21,slist_22,skey_21,skey_22,&slen_21,&slen_22,nn,NULL);
    }
    return;
}

void make_tree(int* list, int* klist,int len, qtree* tree)
{
    int* slist_1=NULL,* slist_2=NULL;
    int* skey_1=NULL,* skey_2=NULL;
    int slen_1, slen_2;
    int nn=0;
    int head_p;
    tree_node* node_p=NULL;
    tree->node_list=(tree_node*)malloc(sizeof(tree_node)*len);
    node_p=tree->head;
    make_tree_node_son(tree->node_list,list,klist,len,&node_p,
    slist_1,slist_2,skey_1,skey_2,&slen_1,&slen_2,&nn,&head_p);
    tree->node_num=len;
    tree->head=&tree->node_list[head_p];
    return;
}
int find_tree_node(int data, tree_node* node)
{
    int result=-1;
    if (node==NULL) return result;
    if (data<node->data) result=find_tree_node(data,node->left_leaf);
    else if (data>node->data) result=find_tree_node(data,node->right_leaf);
    else return node->key;
    return result;
}
int find_tree(int data, qtree* tree)
{
    int result;
    result=find_tree_node(data,tree->head);
    return result;
}
void del_tree_node(qtree* tree)
{
    tree_node* node_p=tree->head;
    tree_node* node_fp=tree->head;
    tree_node* tmp;
    int lr=0;
    if (node_p->left_leaf==NULL && node_p->right_leaf==NULL)
    {
        free(node_p);
        tree->node_num--;
        node_p=NULL;
        return;
    }
    if (node_fp->left_leaf==NULL) {
        node_p=node_fp->right_leaf;
        lr=1;
    }
    else {
        node_p=node_fp->left_leaf;
        lr=-1;
    }
    while (1)
    {
        if (node_p->left_leaf==NULL && node_p->right_leaf==NULL)
        {
            free(node_p);
            if (lr==-1) node_fp->left_leaf=NULL;
            else node_fp->right_leaf=NULL;
            tree->node_num--;
            return;
        }
        tmp=node_p;
        if (node_fp->left_leaf==NULL) {
            node_p=node_fp->right_leaf;
            lr=1;
        }
        else {
            node_p=node_fp->left_leaf;
            lr=-1;
        }
        node_fp=tmp;
    }
    return;
}

void del_tree(qtree* tree)
{
    free(tree->node_list);
    return;
}
int bi_compare( const void * a, const void * b)
{
    binary_num* a_b=(binary_num*)a;
    binary_num* b_b=(binary_num*)b;
    return a_b->i-b_b->i;
}
void sort_binary_list(binary_list* bl)
{
    qsort(bl->data,bl->len,sizeof(binary_num),bi_compare);
    return;
}
