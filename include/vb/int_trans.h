#ifndef _BUILDER_INT_TRANS_H_
#define _BUILDER_INT_TRANS_H_
#include "fock/fock.h"
struct IntransInfo
{
    mol_info mol;     // the structure of molecule
    grids_info grids, fgrids; // the grids used for separate density fitting (SDF)
    bas_info aux;     // auxiliary basis function used for RI or SDF
    /**
     * @brief this structure is used for local integral transformation as HAO in VB calculation
     */
    bool if_sub;      // if do the calculation of subsystem
    bool if_final_load, if_final_use; // if final grids is defined and used
    mol_info mol_sub; // the mol_info of subsystem
    int sub_num;      // the number of atoms in subsystem
    List sub_list;    // which atom in the subsystem
    /**
     * @brief information of grids file and auxiliary basis function
     */
    char *grids_fname, *fgrids_fname;
    char *aux_fname;
    Matrix aux_half;
};
typedef struct IntransInfo *intrans_info;
intrans_info init_intrans(const char *grids_fname, const char* fgrids_fname, const char *aux_fname, const mol_info mol);
void set_intrans_final(bool if_final, intrans_info intrans);
// intrans_info init_intrans_sub(int sub_num, List sub_list, const char *grids_fname, const char* fgrids_fname, const char *aux_fname, const mol_info mol);
void del_intrans(intrans_info intrans);
void cal_intrans_iscf5(Tensor4D tuvw, Tensor4D atuv, Tensor4D ptuv, int vir_num, int act_num, const Matrix vir_orb, const Matrix act_orb, intrans_info intrans);
#endif