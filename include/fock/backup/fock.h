#ifndef _BUILDER_FOCK_H_
#define _BUILDER_FOCK_H_
#include "mol/xint.h"
#include "mol/xgrids.h"
typedef enum Jbuilder_type
{
    None_jbuilder,
    RI_jbuilder,
    SemiNumerical_jbuilder,
    THC_jbuilder
} Jbuilder_type;

typedef enum Kbuilder_type
{
    None_kbuilder,
    RI_kbuilder,
    LDF_kbuilder,
    COSX_kbuilder,
    rCOSX_kbuilder
} Kbuilder_type;
struct CoulombInfo
{
    Jbuilder_type type; // default type is None
    bool if_aux, if_grids;
    int batch_size;
    Matrix *J; // Matrix used to store Coulomb matrix
    mol_info mol;
    bas_info aux;
    Matrix aux2e_matrix;
    Matrix o_matrix; // correction matrix for semi-numerical method
    grids_info grids;
    Vector *df_coe;
};
typedef struct CoulombInfo *coulomb_info;
coulomb_info init_coulomb(int batch_size, const mol_info mol);
void set_coulomb_type(Jbuilder_type type, coulomb_info coulomb);
void load_coulomb_aux(const char *aux_fname, coulomb_info coulomb);
void load_coulomb_grids(const char *grids_fname, coulomb_info coulomb);
void del_coulomb(coulomb_info coulomb);
void cal_j_matrix(const Matrix d_matrix, coulomb_info coulomb);
void cal_j_matrix_batch(const bool *if_finnish, const Matrix *d_matrix, coulomb_info coulomb);
void cal_coulomb_interact(Matrix EJ_act, Vector EJ_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self,
                          const int act_num, const int self_num, coulomb_info coulomb);
void cal_electrostatic_interact_grid(Matrix *EJ_act, Vector *EJ_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self,
                                     const int act_num, const int self_num, const List *atm_id, const grids_info grid_in, coulomb_info coulomb);
void cal_coulomb_grad_batch(Vector *e_grad, const Matrix *d_matrix, coulomb_info coulomb);

struct ExchangeInfo
{
    double lr_frac, sr_frac, omega_; // parameters for range-separate calculation
    Kbuilder_type type;              // default type is None
    bool if_aux, if_grids, if_final;
    int batch_size;
    Matrix *K, *K_b; // Matrix used to store exchange matrix
    mol_info mol;
    bas_info aux, aux_final; // auxiliary basis set of GEN-An and GEN-An* is used
    Matrix aux2e_matrix_rev, aux2e_matrix_rev_final;
    Matrix o_matrix; // correction matrix for semi-numerical method
    grids_info grids;
    Matrix o_matrix_final;
    grids_info grids_final; // if final integral is done
    char *grids_fname;
};
typedef struct ExchangeInfo *exchange_info;
exchange_info init_exchange(int batch_size, const mol_info mol);
void set_exchange_type(Kbuilder_type type, exchange_info exchange);
void set_exchange_rs(double lr_frac, double sr_frac, double omega_, exchange_info exchange);
void set_exchange_final(bool if_final, exchange_info exchange);
void load_exchange_aux(exchange_info exchange);
void load_exchange_grids(const char *grids_fname, const char *fgrids_fname, exchange_info exchange);
void del_exchange(exchange_info exchange);
void cal_k_matrix(const Matrix d_matrix, const Matrix d_matrix_b, exchange_info exchange);
void cal_k_matrix_batch(const bool *if_finnish, const Matrix *d_matrix, const Matrix *d_matrix_b, exchange_info exchange);
void cal_exchange_interact(Matrix EK_act, Vector EK_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self, const Matrix *d_matrix_act_b,
                           const Matrix *d_matrix_self_b, const int act_num, const int self_num, exchange_info fock);
void cal_exchange_interact_grid(Matrix *EK_act, Vector *EK_self, const Matrix *d_matrix_act, const Matrix *d_matrix_self,
                                const Matrix *d_matrix_act_b, const Matrix *d_matrix_self_b, const int act_num, const int self_num, grids_info grids, exchange_info exchange);
void cal_exchange_grad_cosx(Vector *e_grad, const Matrix *d_matrix, const Matrix *d_matrix_b, exchange_info exchange);
#endif
