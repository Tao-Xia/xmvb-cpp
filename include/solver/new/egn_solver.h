#ifndef _DIAG_solver_H_
#define _DIAG_solver_H_
#include "mol/mol.h"
typedef struct DavidsonInfo* davidson_info;
typedef void (*solver_info)(int batch_size, Vector* result, const Vector* basis, void* info);
typedef void (*solver_info_nh)(int batch_size, Vector* result_1, Vector* result_2, const Vector* basis, void* info);
davidson_info init_davidson(int state_num, int msize);
// the integer "num" must bigger than 2*davidson->state_num
void load_naive_davidson(Vector *guess, davidson_info davidson);
void load_guess_davidson(const Vector perturb_vector, davidson_info davidson);
void del_davidson(davidson_info davidson);
Matrix get_davidson_egnvec(const davidson_info davidson);
Vector get_davidson_omega(const davidson_info davidson);
// hermit type as CIS and TDA
void do_davidson_tda(void* info, solver_info solver, davidson_info davidson);
// no hermit type as TDHF and TDDFT
void do_davidson_rpa(void *info, solver_info solver, davidson_info davidson);
// Lanczos for Hermit matrix's egnivalue problem
typedef struct LanczosSolver* lanczos_solver;
typedef void (*Hpsi_builder)(Vector Hpsi, const Vector psi, void* info); 
lanczos_solver init_lanczos(const int eig_num, const int size);
void del_lanczos(lanczos_solver lanczos);
void do_lanczos(const Vector psi, void* info, Hpsi_builder builder, lanczos_solver lanczos);
void set_thrE_lanczos(const double thr_e, lanczos_solver lanczos);
double get_eig_value_lanczos(const int num, const lanczos_solver lanczos);
Vector get_eig_vector_lanczos(const int num, const lanczos_solver lanczos);
#endif
