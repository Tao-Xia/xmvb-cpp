#ifndef _SYMM_H_
#define _SYMM_H_
#include "scf/hf.h"
#include "symm/symm.h"

#define COORD_ERR    1e-4

struct character_table {
  // C1
  double table_c1;
  int index_c1;
  int product_c1_1;
  int product_c1_2;

  // C2
  double table_c2[2][2];
  int index_c2[2];
  int product_c2_1[2][2];
  int product_c2_2[2][2];
//  double table_c2[2][2]={
//    1.0, 1.0,
//    1.0,-1.0
//  };
//  int index_c2[2]={1,2};
//  int product_c2_1[2][2]={
//    1,1,
//    1,0
//  };
//  int product_c2_2[2][2]={
//    1,3,
//    1,0
//  };

  // C3

  // C2v
  double table_c2v[4][4];
  int index_c2v[4];
  int product_c2v_1[4][4];
  int product_c2v_2[4][4];
//  double table_c2v[4][4]= {
//    1.0, 1.0, 1.0, 1.0,
//    1.0, 1.0,-1.0,-1.0,
//    1.0,-1.0, 1.0,-1.0,
//    1.0,-1.0,-1.0, 1.0
//  };
//  int index_c2v[4]={1,2,3,4};
//  int product_c2v_1[4][4]={
//    1,1,1,1,
//    3,5,3,3,
//    5,0,0,0,
//    7,0,0,0
//  };
//  int product_c2v_2[4][4]={
//    1,3,5,7,
//    3,7,7,5,
//    5,0,0,0,
//    7,0,0,0
//  };

  // C3v
  double table_c3v[3][3];
  int index_c3v[6];
  int product_c3v_1[3][3];
  int product_c3v_2[3][3];
//  double table_c3v[3][3]={
//    1.0, 1.0, 2.0,
//    1.0, 1.0,-1.0,
//    1.0,-1.0, 0.0
//  };
//  int index_c3v[6]={1,2,2,3,3,3};
//  int product_c3v_1[3][3]={
//    1,1,5,
//    3,5,0,
//    5,0,0
//  };
//  int product_c3v_2[3][3]={
//    1,3,5,
//    3,5,0,
//    5,0,0
//  };

  // C4v
  double table_c4v[5][5];
  int index_c4v[8];
  int product_c4v_1[5][5];
  int product_c4v_2[5][5];
//  double table_c4v[5][5]={
//    1.0, 1.0, 1.0, 1.0, 2.0,
//    1.0, 1.0,-1.0,-1.0, 0.0,
//    1.0, 1.0, 1.0, 1.0,-2.0,
//    1.0,-1.0, 1.0,-1.0, 0.0,
//    1.0,-1.0,-1.0, 1.0, 0.0
//  };
//  int index_c4v[8]={1,2,3,2,4,4,5,5};
//  int product_c4v_1[5][5]={
//    1,1,1,1,1,
//    3,5,3,3,3,
//    5,9,9,9,5,
//    7,0,0,0,7,
//    9,0,0,0,0
//  };
//  int product_c4v_2[5][5]={
//    1,3,5,7,9,
//    3,7,7,5,9,
//    5,9,9,9,9,
//    7,0,0,0,9,
//    9,0,0,0,0
//  };

  // C5v
  double table_c5v[4][4];
  int index_c5v[10];
  int product_c5v_1[4][4];
  int product_c5v_2[4][4];
//  double table_c5v[4][4]={
//    1.0, 1.0, 2.0,   2.0,
//    1.0, 1.0, 0.618,-1.618,
//    1.0, 1.0,-1.618, 0.618,
//    1.0,-1.0, 0.0,   0.0
//  };
//  int index_c5v[10]={1,2,3,3,2,4,4,4,4,4};
//  int product_c5v_1[4][4]={
//    1,1,1,1,
//    3,5,3,3,
//    5,7,5,5,
//    7,0,7,5
//  };
//  int product_c5v_2[4][4]={
//    1,3,5,7,
//    3,5,5,7,
//    5,7,7,5,
//    7,0,7,7
//  };

  // C6v
  double table_c6v[6][6];
  int index_c6v[12];
  int product_c6v_1[6][6];
  int product_c6v_2[6][6];
//  double table_c6v[6][6]={
//    1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
//    1.0, 1.0,-1.0,-1.0, 1.0,-1.0,
//    1.0, 1.0, 1.0, 1.0,-1.0,-1.0,
//    1.0, 1.0,-1.0,-1.0,-2.0, 2.0,
//    1.0,-1.0, 1.0,-1.0, 0.0, 0.0,
//    1.0,-1.0,-1.0, 1.0, 0.0, 0.0
//  };
//  int index_c6v[12]={1,2,3,4,3,2,5,5,5,6,6,6};
//  int product_c6v_1[6][6]={
//    1 ,1 ,1 ,1 ,1 ,1 ,
//    3 ,5 ,3 ,3 ,3 ,3 ,
//    5 ,9 ,9 ,9 ,5 ,5 ,
//    7 ,11,0 ,0 ,7 ,7 ,
//    9 ,0 ,0 ,0 ,9 ,9 ,
//    11,0 ,0 ,0 ,0 ,11
//  };
//  int product_c6v_2[6][6]={
//    1 ,3 ,5 ,7 ,9 ,11,
//    3 ,7 ,7 ,5 ,9 ,11,
//    5 ,9 ,11,11,11,9 ,
//    7 ,11,0 ,0 ,11,9 ,
//    9 ,0 ,0 ,0 ,11,9 ,
//    11,0 ,0 ,0 ,0 ,11
//  };

};

struct SymmInfo {
  double inertia[3][3];
  char pointgroup[10];
  int linear,plane,regularpolygon,irregularpolygon,polyhedron;
  int num_axis,num_vector;
  double *vector;
  double *vector2;
  double *axis;
  int *vector_state;
  int *axis_order;
  int *type_axis;
  int *type_plane;

  int *irred;
  int *num_irred;
  int *irred_2e;
  int *num_irred_2e;
  char *name_irred;
  int *state_ij;
  int numirred,numsymm;
  double *table;
  int *index2;
  int *product1,*product2;
  int *nab;
  double *cab,*sab;

  int *state_1e;
  int *state_2e;
  int *num_2e;
};

typedef struct SymmInfo* symm_info;

int detect_symm(mol_info mol, symm_info symm);
int coord_tran(double *coord, double *mass, mol_info mol, symm_info symm);
symm_info init_symm();
void del_symm(symm_info symm);
int detect_pointgroup(double *coord, int *atom_num, mol_info mol, symm_info symm);
int testparallel(double ax, double ay,double az, double nx, double ny, double nz);
int select_symm(mol_info mol,symm_info symm);

#endif
