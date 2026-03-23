#include "mol/mol.h"
static int angular_len[] = {1, 3, 6, 10, 15, 21};
const static char ele_names[] = {' ', ' ',
                                 'H', ' ', 'H', 'e',
                                 'L', 'i', 'B', 'e', 'B', ' ', 'C', ' ', 'N', ' ', 'O', ' ', 'F', ' ', 'N', 'e',
                                 'N', 'a', 'M', 'g', 'A', 'l', 'S', 'i', 'P', ' ', 'S', ' ', 'C', 'l', 'A', 'r',
                                 'K', ' ', 'C', 'a', 'S', 'c', 'T', 'i', 'V', ' ', 'C', 'r', 'M', 'n', 'F', 'e', 'C', 'o', 'N', 'i', 'C', 'u', 'Z', 'n', 'G','a','G', 'e', 'A', 's', 'S', 'e', 'B', 'r', 'K', 'r'};

static double frac(int n)
{
    double result = 1.;
    for (int i = n; n > 0; n--)
        result *= n;
    return result;
}

extern int xint_gtolen(int ang)
{
    assert(ang <= 5);
    return angular_len[ang];
}
extern double xint_norm(int angular, double exp_num)
{
    double norm;
    norm = pow(2., 2 * angular + 3) * pow(2 * exp_num, angular + 1.5) / sqrt(MY_PI);
    norm *= frac(angular + 1);
    norm /= frac(2 * angular + 2);
    norm = sqrt(norm);
    if (angular == 0)
        norm *= NORM_S;
    else if (angular == 1)
        norm *= NORM_P;
    return norm;
}
extern int ele2charge(const char *ele)
{
    int charge = 0;
    if (ele[0] == 'H' && ele[1] == '\0')
        charge = 1;
    else if (ele[0] == 'H' && (ele[1] == 'E' || ele[1] == 'e'))
        charge = 2;
    else if (ele[0] == 'L' && (ele[1] == 'I' || ele[1] == 'i'))
        charge = 3;
    else if (ele[0] == 'B' && (ele[1] == 'E' || ele[1] == 'e'))
        charge = 4;
    else if (ele[0] == 'B' && ele[1] == '\0')
        charge = 5;
    else if (ele[0] == 'C' && ele[1] == '\0')
        charge = 6;
    else if (ele[0] == 'N' && ele[1] == '\0')
        charge = 7;
    else if (ele[0] == 'O' && ele[1] == '\0')
        charge = 8;
    else if (ele[0] == 'F' && ele[1] == '\0')
        charge = 9;
    else if (ele[0] == 'N' && (ele[1] == 'E' || ele[1] == 'e'))
        charge = 10;
    else if (ele[0] == 'N' && (ele[1] == 'A' || ele[1] == 'a'))
        charge = 11;
    else if (ele[0] == 'M' && (ele[1] == 'G' || ele[1] == 'g'))
        charge = 12;
    else if (ele[0] == 'A' && (ele[1] == 'L' || ele[1] == 'l'))
        charge = 13;
    else if (ele[0] == 'S' && (ele[1] == 'I' || ele[1] == 'i'))
        charge = 14;
    else if (ele[0] == 'P' && ele[1] == '\0')
        charge = 15;
    else if (ele[0] == 'S' && ele[1] == '\0')
        charge = 16;
    else if (ele[0] == 'C' && (ele[1] == 'L' || ele[1] == 'l'))
        charge = 17;
    else if (ele[0] == 'A' && (ele[1] == 'R' || ele[1] == 'r'))
        charge = 18;
    else if (ele[0] == 'C' && ((ele[1]) == 'a' || ele[1] == 'A'))
        charge = 20;
    else if (ele[0] == 'S' && ((ele[1]) == 'c' || ele[1] == 'C'))
        charge = 21;
    else if (ele[0] == 'T' && ((ele[1]) == 'i' || ele[1] == 'I'))
        charge = 22;
    else if (ele[0] == 'C' && ((ele[1]) == 'r' || ele[1] == 'R'))
        charge = 24;
    else if (ele[0] == 'M' && ((ele[1]) == 'n' || ele[1] == 'N'))
        charge = 25;
    else if (ele[0] == 'F' && ((ele[1]) == 'e' || ele[1] == 'E'))
        charge = 26;
    else if (ele[0] == 'C' && ((ele[1]) == 'o' || ele[1] == 'O'))
        charge = 27;
    else if (ele[0] == 'N' && ((ele[1]) == 'i' || ele[1] == 'I'))
        charge = 28;
    else if (ele[0] == 'C' && ((ele[1]) == 'u' || ele[1] == 'U'))
        charge = 29;
    else if (ele[0] == 'Z' && ((ele[1]) == 'n' || ele[1] == 'N'))
        charge = 30;
    else if (ele[0] == 'G' && ((ele[1]) == 'a' || ele[1] == 'A'))
        charge = 31;
    else if (ele[0] == 'G' && ((ele[1]) == 'e' || ele[1] == 'E'))
        charge = 32;
    else if (ele[0] == 'A' && ((ele[1]) == 's' || ele[1] == 'S'))
        charge = 33;
    else if (ele[0] == 'S' && ((ele[1]) == 'e' || ele[1] == 'E'))
        charge = 34;
    else if (ele[0] == 'B' && ((ele[1]) == 'r' || ele[1] == 'R'))
        charge = 35;
    else if (ele[0] == 'K' && ((ele[1]) == 'r' || ele[1] == 'R'))
        charge = 36;
    else if (ele[0] == 'R' && ((ele[1]) == 'b' || ele[1] == 'B'))
        charge = 37;
    else if (ele[0] == 'S' && ((ele[1]) == 'r' || ele[1] == 'R'))
        charge = 38;
    else if (ele[0] == 'Z' && ((ele[1]) == 'r' || ele[1] == 'R'))
        charge = 40;
    else if (ele[0] == 'N' && ((ele[1]) == 'b' || ele[1] == 'B'))
        charge = 41;
    else if (ele[0] == 'M' && ((ele[1]) == 'o' || ele[1] == 'O'))
        charge = 42;
    else if (ele[0] == 'T' && ((ele[1]) == 'c' || ele[1] == 'C'))
        charge = 43;
    else if (ele[0] == 'R' && ((ele[1]) == 'u' || ele[1] == 'U'))
        charge = 44;
    else if (ele[0] == 'R' && ((ele[1]) == 'h' || ele[1] == 'H'))
        charge = 45;
    else if (ele[0] == 'P' && ((ele[1]) == 'd' || ele[1] == 'D'))
        charge = 46;
    else if (ele[0] == 'A' && ((ele[1]) == 'g' || ele[1] == 'G'))
        charge = 47;
    else if (ele[0] == 'C' && ((ele[1]) == 'd' || ele[1] == 'D'))
        charge = 48;
    else if (ele[0] == 'I' && ((ele[1]) == 'n' || ele[1] == 'N'))
        charge = 49;
    else if (ele[0] == 'S' && ((ele[1]) == 'n' || ele[1] == 'N'))
        charge = 50;
    else if (ele[0] == 'S' && ((ele[1]) == 'b' || ele[1] == 'B'))
        charge = 51;
    else if (ele[0] == 'T' && ((ele[1]) == 'e' || ele[1] == 'E'))
        charge = 52;
    else if (ele[0] == 'X' && ((ele[1]) == 'e' || ele[1] == 'E'))
        charge = 54;
    else if (ele[0] == 'C' && ((ele[1]) == 's' || ele[1] == 'S'))
        charge = 55;
    else if (ele[0] == 'B' && ((ele[1]) == 'a' || ele[1] == 'A'))
        charge = 56;
    else if (ele[0] == 'L' && ((ele[1]) == 'a' || ele[1] == 'A'))
        charge = 57;
    else if (ele[0] == 'C' && ((ele[1]) == 'e' || ele[1] == 'E'))
        charge = 58;
    else if (ele[0] == 'P' && ((ele[1]) == 'r' || ele[1] == 'R'))
        charge = 59;
    else if (ele[0] == 'N' && ((ele[1]) == 'd' || ele[1] == 'D'))
        charge = 60;
    else if (ele[0] == 'P' && ((ele[1]) == 'm' || ele[1] == 'M'))
        charge = 61;
    else if (ele[0] == 'S' && ((ele[1]) == 'm' || ele[1] == 'M'))
        charge = 62;
    else if (ele[0] == 'E' && ((ele[1]) == 'u' || ele[1] == 'U'))
        charge = 63;
    else if (ele[0] == 'G' && ((ele[1]) == 'd' || ele[1] == 'D'))
        charge = 64;
    else if (ele[0] == 'T' && ((ele[1]) == 'b' || ele[1] == 'B'))
        charge = 65;
    else if (ele[0] == 'D' && ((ele[1]) == 'y' || ele[1] == 'Y'))
        charge = 66;
    else if (ele[0] == 'H' && ((ele[1]) == 'o' || ele[1] == 'O'))
        charge = 67;
    else if (ele[0] == 'E' && ((ele[1]) == 'r' || ele[1] == 'R'))
        charge = 68;
    else if (ele[0] == 'T' && ((ele[1]) == 'm' || ele[1] == 'M'))
        charge = 69;
    else if (ele[0] == 'Y' && ((ele[1]) == 'b' || ele[1] == 'B'))
        charge = 70;
    else if (ele[0] == 'L' && ((ele[1]) == 'u' || ele[1] == 'U'))
        charge = 71;
    else if (ele[0] == 'H' && ((ele[1]) == 'f' || ele[1] == 'F'))
        charge = 72;
    else if (ele[0] == 'T' && ((ele[1]) == 'a' || ele[1] == 'A'))
        charge = 73;
    else if (ele[0] == 'R' && ((ele[1]) == 'e' || ele[1] == 'E'))
        charge = 75;
    else if (ele[0] == 'O' && ((ele[1]) == 's' || ele[1] == 'S'))
        charge = 76;
    else if (ele[0] == 'I' && ((ele[1]) == 'r' || ele[1] == 'R'))
        charge = 77;
    else if (ele[0] == 'P' && ((ele[1]) == 't' || ele[1] == 'T'))
        charge = 78;
    else if (ele[0] == 'A' && ((ele[1]) == 'u' || ele[1] == 'U'))
        charge = 79;
    else if (ele[0] == 'H' && ((ele[1]) == 'g' || ele[1] == 'G'))
        charge = 80;
    else if (ele[0] == 'T' && ((ele[1]) == 'l' || ele[1] == 'L'))
        charge = 81;
    else if (ele[0] == 'P' && ((ele[1]) == 'b' || ele[1] == 'B'))
        charge = 82;
    else if (ele[0] == 'B' && ((ele[1]) == 'i' || ele[1] == 'I'))
        charge = 83;
    else if (ele[0] == 'P' && ((ele[1]) == 'o' || ele[1] == 'O'))
        charge = 84;
    else if (ele[0] == 'A' && ((ele[1]) == 't' || ele[1] == 'T'))
        charge = 85;
    else if (ele[0] == 'R' && ((ele[1]) == 'n' || ele[1] == 'N'))
        charge = 86;
    else if (ele[0] == 'K')
        charge = 19;
    else if (ele[0] == 'V')
        charge = 23;
    else if (ele[0] == 'Y')
        charge = 39;
    else if (ele[0] == 'I')
        charge = 53;
    return charge;
}
char *charge2ele(int charge)
{
    char *ele = NULL;
    ele = (char *)calloc(sizeof(char), 4);
    ele[0] = ele_names[2 * charge];
    ele[1] = ele_names[2 * charge + 1];
    return ele;
}
extern int ang2num(const char *ang)
{
    if (ang[0] == 'S' && ang[1] == 'P')
        return -1;
    else if (ang[0] == 'S')
        return 0;
    else if (ang[0] == 'P')
        return 1;
    else if (ang[0] == 'D')
        return 2;
    else if (ang[0] == 'F')
        return 3;
    else if (ang[0] == 'G')
        return 4;
    else if (ang[0] == 'H')
        return 5;
    return 0;
}

atm_info init_atm(const char *mol_fname,int unit_bohr)
{
    int qoff = 0, e_num = 0;
    char string[1024]; /// buf used to store file's words
    char tmp1[64], tmp2[64], tmp3[64], tmp4[64];
    char *err;
    FILE *mol_file = fopen(mol_fname, "r");
    atm_info atm = (atm_info)malloc(sizeof(struct AtmInfo));
    assert(mol_file != NULL);
    /// read in information of molecule
    err = fgets(&string[0], sizeof(string), mol_file);
    err = fgets(&string[0], sizeof(string), mol_file);
    err = fgets(&string[0], sizeof(string), mol_file);
    err = fgets(&string[0], sizeof(string), mol_file);
    string[strlen(string) - 1] = '\0';
    atm->natm = atoi(string); /// the first line of .MOL is the number of atoms
    atm->atm = (int *)malloc(sizeof(int) * 2 * atm->natm);
    err = fgets(&string[0], sizeof(string), mol_file);
    string[strlen(string) - 1] = '\0';
    sscanf(string, "%s%s", tmp1, tmp2);
    atm->charge = atoi(tmp1);
    atm->mult = atoi(tmp2);
    atm->value = malloc_vector(3 * atm->natm);
    for (int i = 0; i < atm->natm; i++)
    {
        err = fgets(&string[0], sizeof(string), mol_file);
        sscanf(string, "%s%s%s%s", tmp1, tmp2, tmp3, tmp4);
        atm(ELEMENT_VAL, i) = ele2charge(tmp1);
        atm(CENTER_IND, i) = qoff;
        if (unit_bohr==0) {
          atm->value[qoff + 0] = atof(tmp2) / A2AU;
          atm->value[qoff + 1] = atof(tmp3) / A2AU;
          atm->value[qoff + 2] = atof(tmp4) / A2AU;
        }
        else {
          atm->value[qoff + 0] = atof(tmp2);
          atm->value[qoff + 1] = atof(tmp3);
          atm->value[qoff + 2] = atof(tmp4);
        }
        qoff += 3;
    }
    fclose(mol_file);
    for (int i = 0; i < atm->natm; i++)
        e_num += atm(ELEMENT_VAL, i);
    e_num -= atm->charge;
    e_num += (atm->mult - 1);
    atm->alpha_num = e_num / 2;
    e_num -= (atm->mult - 1);
    atm->beta_num = e_num - atm->alpha_num;
    atm->mol_fname = (char *)malloc(sizeof(char) * (strlen(mol_fname) + 4) / 4 * 4);
    memset(atm->mol_fname, 0, sizeof(char) * (strlen(mol_fname) + 4) / 4 * 4);
    memcpy(atm->mol_fname, mol_fname, sizeof(char) * strlen(mol_fname));
#ifdef DEBUG
    printf("  File Name: %s[%ld]\n", mol_fname, (strlen(mol_fname) + 4) / 4 * 4);
    printf("  The charge is %2d and multi spin is %2d.\n", atm->charge, atm->mult);
    printf("  The alpha electron's number is %5d and beta electron's number is %5d.\n", atm->alpha_num, atm->beta_num);
    printf("  The structure of molecule is shown as:\n");
    print_atm(atm);
#endif
    return atm;
}

void del_atm(atm_info atm)
{
    free(atm->atm);
    free(atm->value);
    free(atm->mol_fname);
    free(atm);
    return;
}

void print_atm(const atm_info atm)
{
    int natm = atm->natm;
    for (int i = 0; i < natm; i++)
    {
        char *ele = charge2ele(atm(ELEMENT_VAL, i));
        printf("  %s %3d %12.5E %12.5E %12.5E\n", ele, atm(ELEMENT_VAL, i),
               atm->value[atm(CENTER_IND, i) + 0], atm->value[atm(CENTER_IND, i) + 1], atm->value[atm(CENTER_IND, i) + 2]);
        free(ele);
    }
    return;
}

inline Vector get_atm_coord(int a, const atm_info atm)
{
    return &atm->value[atm(CENTER_IND, a)];
}

atm_info get_atm_seg(int charge, int mult, int begin_id, int end_id, const atm_info atm)
{
    atm_info atm_seg = (atm_info)malloc(sizeof(struct AtmInfo));
    atm_seg->natm = end_id - begin_id;
    atm_seg->charge = charge;
    atm_seg->mult = mult;
    atm_seg->atm = malloc_list(2 * atm_seg->natm);
    atm_seg->value = malloc_vector(3 * atm_seg->natm);
    int e_num = charge;
    for (int i = begin_id, j = 0; i < end_id; i++, j++)
    {
        atm_seg->atm[j * 2 + ELEMENT_VAL] = atm(ELEMENT_VAL, i);
        atm_seg->atm[j * 2 + CENTER_IND] = 3 * j;
        copy_vector(atm_seg->value + 3 * j, get_atm_coord(i, atm), 3);
        e_num += atm(ELEMENT_VAL, i);
    }
    e_num += (atm_seg->mult - 1);
    atm_seg->alpha_num = e_num / 2;
    e_num -= (atm_seg->mult - 1);
    atm_seg->beta_num = e_num - atm_seg->alpha_num;
    atm_seg->mol_fname = NULL;
    return atm_seg;
}

bas_info init_bas(const char *bas_fname, const atm_info atm)
{
#define bas_tmp(parameter, index) bas_tmp[(index) * 5 + parameter]
    int qoff = 0, e_num = 0;
    char string[1024]; /// buf used to store file's words
    char tmp1[64], tmp2[64], tmp3[64], tmp4[64];
    char *err;
    const char *bas_head = "****\0\0\0";
    int bas_tmp[5 * 1024 * 16], nbas = 0, prim_num;
    double value[1024 * 16];
    int ele_list[108], ele_num = 0, if_in;
    int ele_num_list[108] = {0};
    int ele_tmp;
    int ang;
    FILE *bas_file = fopen(bas_fname, "r");
    if (bas_file == NULL) {
        char *errmessage = (char *)malloc(strlen("Failed to open the basis file ") + strlen(bas_fname) + 1);
        strcpy(errmessage, "Failed to open the basis file ");
        strcat(errmessage, bas_fname);
        perror(errmessage);
        free(errmessage);
        exit(1); 
    }
    bas_info bas = (bas_info)malloc(sizeof(struct BasInfo));
    // assert(bas_file != NULL);
    bas->atm = atm;
    /// get the different type elements
    for (int i = 0; i < atm->natm; i++)
    {
        if_in = 0;
        for (int j = 0; j < ele_num; j++)
            if (atm(ELEMENT_VAL, i) == ele_list[j])
            {
                if_in = 1;
                ele_num_list[j]++;
                break;
            }
        if (if_in == 0)
        {
            ele_list[ele_num] = atm(ELEMENT_VAL, i);
            ele_num_list[ele_num]++;
            ele_num++;
        }
    }
    /// get the basis set of each element
    while (err = fgets(&string[0], sizeof(string), bas_file))
    {
        if (strstr(string, bas_head) != NULL)
        {
        GET_BAS:
            err = fgets(string, sizeof(string), bas_file);
            if (strlen(string) <= 1)
                break;
            sscanf(string, "%s%s", tmp1, tmp2);
            ele_tmp = ele2charge(tmp1);
            if_in = 0;
            for (int i = 0; i < ele_num; i++)
                if (ele_list[i] == ele_tmp)
                {
                    if_in = 1;
                    break;
                }
            if (if_in)
            {
                while (1)
                {
                    err = fgets(string, sizeof(string), bas_file);
                    if (strstr(string, bas_head) != NULL)
                        goto GET_BAS;
                    sscanf(string, "%s%s%s", tmp1, tmp2, tmp3);
                    prim_num = atoi(tmp2);
                    ang = ang2num(tmp1);
                    if (ang != -1)
                    {
                        bas_tmp(0, nbas) = ele_tmp;
                        bas_tmp(ANGULAR_VAL, nbas) = ang2num(tmp1);
                        bas_tmp(NPRIM_VAL, nbas) = prim_num;
                        bas_tmp(EXP_IND, nbas) = qoff;
                        bas_tmp(COEFF_IND, nbas) = qoff + prim_num;
                        for (int i = 0; i < prim_num; i++)
                        {
                            err = fgets(string, sizeof(string), bas_file);
                            sscanf(string, "%s%s", tmp1, tmp2);
                            value[qoff] = atof(tmp1);
                            value[qoff + prim_num] = atof(tmp2) * xint_norm(ang, atof(tmp1));
                            qoff++;
                        }
                        nbas++;
                        qoff += prim_num;
                    }
                    else
                    {
                        bas_tmp(0, nbas) = ele_tmp;
                        bas_tmp(ANGULAR_VAL, nbas) = 0;
                        bas_tmp(NPRIM_VAL, nbas) = prim_num;
                        bas_tmp(EXP_IND, nbas) = qoff;
                        bas_tmp(COEFF_IND, nbas) = qoff + prim_num;
                        bas_tmp(0, nbas + 1) = ele_tmp;
                        bas_tmp(ANGULAR_VAL, nbas + 1) = 1;
                        bas_tmp(NPRIM_VAL, nbas + 1) = prim_num;
                        bas_tmp(EXP_IND, nbas + 1) = qoff + prim_num * 2;
                        bas_tmp(COEFF_IND, nbas + 1) = qoff + prim_num * 3;
                        for (int i = 0; i < prim_num; i++)
                        {
                            err = fgets(string, sizeof(string), bas_file);
                            sscanf(string, "%s%s%s", tmp1, tmp2, tmp3);
                            value[qoff + prim_num * 0] = atof(tmp1);
                            value[qoff + prim_num * 1] = atof(tmp2) * xint_norm(0, atof(tmp1));
                            value[qoff + prim_num * 2] = atof(tmp1);
                            value[qoff + prim_num * 3] = atof(tmp3) * xint_norm(1, atof(tmp1));
                            qoff++;
                        }
                        nbas += 2;
                        qoff += 4 * prim_num;
                    }
                }
            }
        }
    }
    fclose(bas_file);
    /// get basis function of each atom
    bas->nbas = 0;
    bas->value = malloc_vector(qoff);
    memcpy(bas->value, value, sizeof(double) * qoff);
    for (int i = 0; i < ele_num; i++)
    {
        for (int j = 0; j < nbas; j++)
        {
            if (bas_tmp(0, j) == ele_list[i])
            {
                bas->nbas += ele_num_list[i];
            }
        }
    }
    bas->bas = (int *)malloc(sizeof(int) * 5 * bas->nbas);
    bas->shls_p = (int *)malloc(sizeof(int) * bas->nbas);
    bas->nbas = 0.;
    for (int i = 0; i < atm->natm; i++)
    {
        for (int j = 0; j < nbas; j++)
        {
            if (atm(ELEMENT_VAL, i) == bas_tmp(ATOM_IND, j))
            {
                bas(ATOM_IND, bas->nbas) = i;
                bas(ANGULAR_VAL, bas->nbas) = bas_tmp(ANGULAR_VAL, j);
                bas(NPRIM_VAL, bas->nbas) = bas_tmp(NPRIM_VAL, j);
                bas(EXP_IND, bas->nbas) = bas_tmp(EXP_IND, j);
                bas(COEFF_IND, bas->nbas) = bas_tmp(COEFF_IND, j);
                bas->nbas++;
            }
        }
    }
    // sort_bas(mol);  // resort the basis to let basis with larger angular have smaller index
    bas->msize = 0.;
    for (int i = 0; i < bas->nbas; i++)
    {
        bas->shls_p[i] = bas->msize;
        bas->msize += angular_len[bas(ANGULAR_VAL, i)];
    }
    bas->qoff = qoff;
#undef bas_tmp
    bas->bas_fname = (char *)malloc(sizeof(char) * (strlen(bas_fname) + 4) / 4 * 4);
    memset(bas->bas_fname, 0, (strlen(bas_fname) + 4) / 4 * 4);
    memcpy(bas->bas_fname, bas_fname, sizeof(char) * strlen(bas_fname));
#ifdef DEBUG
    printf("  The number of shells is %5d and the number of basis functions is %5d\n", bas->nbas, bas->msize);
    print_bas(bas);
#endif
    return bas;
}

/**
 * @brief generate auxiliary basis as GEN-An or GEN-An*
 *        DOI: 10.1063/1.2431643
 * @param lev 2-4 for 'n' in GEN-An
 * @param if_star if spdf-type auxiliary basis function is used as GEM-An*
 * @param bas original basis function
 * @return bas_info auxiliary basis function
 */
bas_info init_aux(int lev, bool if_star, const bas_info bas)
{
    const atm_info atm = bas->atm;
    bas_info aux = (bas_info)malloc(sizeof(struct BasInfo));
    int nbas = bas->nbas;
    int natm = atm->natm;
    Vector max_exp, min_exp;
    max_exp = malloc_vector(natm);
    min_exp = malloc_vector(natm);
    for (int i = 0; i < natm; i++)
        min_exp[i] = 1E7;
    for (int i = 0; i < nbas; i++)
    {
        int atm_id = bas(ATOM_IND, i);
        int nprm = bas(NPRIM_VAL, i);
        for (int j = 0; j < nprm; j++)
        {
            double exp_val = bas->value[bas(EXP_IND, i) + j];
            if (exp_val > max_exp[atm_id])
                max_exp[atm_id] = exp_val;
            if (exp_val < min_exp[atm_id])
                min_exp[atm_id] = exp_val;
        }
    }
    // generate auxiliary basis function
    int naux = 0;
    int aux_tmp[1024 * 64 * 5];
    double value_aux[1024 * 64];
    int qoff = 1;
    value_aux[0] = 1.;
#define aux_tmp(parameter, index) aux_tmp[(index) * 5 + parameter]
#define append_aux(a)               \
    aux_tmp(NPRIM_VAL, naux) = 1.;  \
    aux_tmp(ANGULAR_VAL, naux) = a; \
    aux_tmp(EXP_IND, naux) = qoff;  \
    aux_tmp(ATOM_IND, naux) = i;    \
    aux_tmp(COEFF_IND, naux) = 0;   \
    naux++;
    for (int i = 0; i < natm; i++)
    {
        int N = my_min(my_max((int)(log(max_exp[i] / min_exp[i]) / log(6. - lev) + 0.5), 3), 5); //(A1)
        /**
         * @brief generate auxiliary basis function of s-type
         */
        double exp_0 = 2 * min_exp[i] * pow(6 - lev, N - 1); // (A2)
        double exp_1 = (1 + lev / (12 - 2. * lev)) * exp_0;  // (A3)
        double exp_2 = exp_0 / (6 - lev);                    // (A4)
        value_aux[qoff] = exp_1;
        append_aux(0);
        qoff++;
        value_aux[qoff] = exp_2;
        append_aux(0);
        qoff++;
        for (int j = 2; j < N; j++)
        {
            append_aux(0);
            value_aux[qoff] = value_aux[qoff - 1] / (6 - lev); // (A5)
            qoff++;
        }
        /**
         * @brief generate auxiliary basis function of spd-type
         *        H, He only have s-type
         */
        if (atm(ELEMENT_VAL, i) <= 2)
            continue;
        exp_0 = value_aux[qoff - 1] / (6 - lev);
        exp_1 = (1 + lev / (12 - 2. * lev)) * exp_0;
        exp_2 = exp_0 / (6 - lev);
        value_aux[qoff] = exp_1;
        append_aux(0);
        append_aux(1);
        append_aux(2);
        qoff++;
        value_aux[qoff] = exp_2;
        append_aux(0);
        append_aux(1);
        append_aux(2);
        qoff++;
        N = my_max(N - 1, 3);
        for (int j = 2; j < N; j++)
        {
            append_aux(0);
            append_aux(1);
            append_aux(2);
            value_aux[qoff] = value_aux[qoff - 1] / (6 - lev); // (A5)
            qoff++;
        }
        /**
         * @brief generate spdf-type auxiliary basis function for GEN-An*
         */
        exp_0 = value_aux[qoff - 1] / (6 - lev);
        exp_1 = (1 + lev / (12 - 2. * lev)) * exp_0;
        exp_2 = exp_0 / (6 - lev);
        value_aux[qoff] = exp_1;
        append_aux(0);
        append_aux(1);
        append_aux(2);
        append_aux(3);
        qoff++;
        value_aux[qoff] = exp_2;
        append_aux(0);
        append_aux(1);
        append_aux(2);
        append_aux(3);
        qoff++;
        N = my_max(N - 1, 3);
        for (int j = 2; j < N; j++)
        {
            value_aux[qoff] = value_aux[qoff - 1] / (6 - lev); // (A5)
            append_aux(0);
            append_aux(1);
            append_aux(2);
            append_aux(3);
            qoff++;
        }
    }
    free(max_exp);
    free(min_exp);
#undef tmp_aux
#undef append_aux
    aux->nbas = naux;
    aux->atm = atm;
    aux->bas_fname = NULL;
    aux->value = malloc_vector(qoff);
    aux->bas = malloc_list(naux * 5);
    aux->shls_p = malloc_list(naux);
    aux->msize = 0;
    memcpy(aux->value, value_aux, sizeof(double) * qoff);
    memcpy(aux->bas, aux_tmp, sizeof(int) * 5 * naux);
    for (int i = 0; i < naux; i++)
    {
        aux->shls_p[i] = aux->msize;
        aux->msize += xint_gtolen(aux(ANGULAR_VAL, i));
    }
    aux->qoff = qoff;
    return aux;
}
void del_bas(bas_info bas)
{
    free(bas->bas);
    free(bas->shls_p);
    free(bas->value);
    free(bas->bas_fname);
    free(bas);
    return;
}
extern double cal_bas_extent(int shli, const bas_info bas)
{
    double *exp_coe, *comb_coe;
    double r_1 = 2, r_2;
    double delta = 1E-10;
    double r_c;
    int len = bas(NPRIM_VAL, shli);
    int l = bas(ANGULAR_VAL, shli);
    exp_coe = &bas->value[bas(EXP_IND, shli)];
    comb_coe = &bas->value[bas(COEFF_IND, shli)];
    do
    {
        double basf = 0.;
        for (int i = 0; i < len; i++)
        {
            basf += fabs(comb_coe[i]) * exp(-r_1 * r_1 * exp_coe[i]) * pow(r_1, l);
        }
        if (basf <= delta)
        {
            break;
        }
        r_1 *= 2.;
    } while (1);
    r_2 = r_1;
    do
    {
        double basf = 0.;
        for (int i = 0; i < len; i++)
        {
            basf += fabs(comb_coe[i]) * exp(-r_2 * r_2 * exp_coe[i]) * pow(r_2, l);
        }
        if (basf >= delta)
            break;
        r_2 *= 0.5;
    } while (1);
    do
    {
        r_c = 0.5 * (r_1 + r_2);
        double basf = 0.;
        for (int i = 0; i < len; i++)
        {
            basf += fabs(comb_coe[i]) * exp(-r_c * r_c * exp_coe[i]) * pow(r_c, l);
        }
        if (basf < delta)
            r_1 = r_c;
        else
            r_2 = r_c;
    } while (r_1 - r_2 > 1E-8);
    return r_c;
}
void print_bas(const bas_info bas)
{
    int nbas = bas->nbas;
    atm_info atm = bas->atm;
    Vector value = bas->value;
    for (int i = 0; i < nbas; i++)
    {
        printf("  ======================< %-5d >====================\n", i + 1);
        int nprim = bas(NPRIM_VAL, i);
        char *ele = charge2ele(atm(ELEMENT_VAL, bas(ATOM_IND, i)));
        printf("  %5d%s::%2d\n", bas(ATOM_IND, i) + 1, ele, bas(ANGULAR_VAL, i));
        for (int j = 0; j < nprim; j++)
        {
            printf("  %3d: %2.10E           %2.10E\n", j + 1, value[bas(EXP_IND, i) + j],
                   value[bas(COEFF_IND, i) + j] / xint_norm(bas(ANGULAR_VAL, i), value[bas(EXP_IND, i) + j]));
        }
        free(ele);
    }
    printf("  ===================================================\n");
    return;
}
inline Vector get_bas_exp(int shli, const bas_info bas)
{
    return &bas->value[bas(EXP_IND, shli)];
}
inline Vector get_bas_coe(int shli, const bas_info bas)
{
    return &bas->value[bas(COEFF_IND, shli)];
}
prm_info init_prm(const bas_info bas)
{
#define prm_tmp(parameter, index) prm_tmp[(index) * 3 + parameter]
    int nbas = bas->nbas;
    const Vector value_bas = bas->value;
    double value_tmp[1024 * 16] = {0.};
    int prm_tmp[1024 * 16] = {0};
    prm_info prm = (prm_info)malloc(sizeof(struct PrmInfo));
    prm->atm = bas->atm;
    prm->bas = bas;
    prm->nprm = 0;
    prm->c2p_len = 0;
    prm->msize = 0;
    for (int i = 0; i < nbas; i++)
    {
        int prim_num = bas(NPRIM_VAL, i);
        int atm_id = bas(ATOM_IND, i);
        const Vector exp_coe = &value_bas[bas(EXP_IND, i)];
        prm->c2p_len += prim_num;
        for (int j = 0; j < prim_num; j++)
        {
            bool if_in = false;
            for (int k = 0; k < prm->nprm; k++)
            {
                if (value_tmp[prm_tmp(EXP_IND, k)] == exp_coe[j] && atm_id == prm_tmp(ATOM_IND, k))
                {
                    if_in = true;
                    break;
                }
            }
            if (!if_in)
            {
                prm_tmp(ATOM_IND, prm->nprm) = atm_id;
                prm_tmp(ANGULAR_VAL, prm->nprm) = bas(ANGULAR_VAL, i);
                prm_tmp(EXP_IND, prm->nprm) = prm->nprm;
                value_tmp[prm->nprm] = exp_coe[j];
                prm->nprm++;
                prm->msize += xint_gtolen(bas(ANGULAR_VAL, i));
            }
        }
    }
    prm->prm = malloc_list(3 * prm->nprm);
    prm->value = malloc_vector(prm->nprm);
    prm->shls_p = malloc_list(prm->nprm);
    prm->c2p_coe = malloc_vector(prm->c2p_len);
    prm->c2p_index = malloc_list(prm->c2p_len * 2);
    memcpy(prm->value, value_tmp, sizeof(double) * prm->nprm);
    memcpy(prm->prm, prm_tmp, sizeof(int) * prm->nprm * 3);
    for (int i = 1; i < prm->nprm; i++)
        prm->shls_p[i] = prm->shls_p[i - 1] + xint_gtolen(prm(ANGULAR_VAL, i));
    for (int i = 0, uu = 0; i < nbas; i++)
    {
        int prim_num = bas(NPRIM_VAL, i);
        int atm_id = bas(ATOM_IND, i);
        const Vector exp_coe = &value_bas[bas(EXP_IND, i)];
        for (int j = 0; j < prim_num; j++)
        {
            for (int k = 0; k < prm->nprm; k++)
            {
                if (prm->value[prm(EXP_IND, k)] == exp_coe[j] && atm_id == prm(ATOM_IND, k))
                {
                    prm->c2p_index[uu * 2 + 0] = i;
                    prm->c2p_index[uu * 2 + 1] = k;
                    prm->c2p_coe[uu] = value_bas[bas(COEFF_IND, i) + j];
                    uu++;
                    break;
                }
            }
        }
    }
#undef prm_tmp
#ifdef DEBUG
    printf("  The number of prime shells is %5d and the number of prime basis function s %5d\n", prm->nprm, prm->msize);
    printf("     ID  ATOM      ANGULAR   EXP COEFFICIENT\n");
    for (int i = 0; i < prm->nprm; i++)
    {
        char *ele = charge2ele(bas->atm(ELEMENT_VAL, prm(ATOM_IND, i)));
        printf("  %5d  %-2s[%4d]    %3d     %12.6f\n", i + 1, ele, prm(ATOM_IND, i), prm(ANGULAR_VAL, i), prm->value[prm(EXP_IND, i)]);
        free(ele);
    }
#endif
    return prm;
}

void del_prm(prm_info prm)
{
    free(prm->value);
    free(prm->shls_p);
    free(prm->prm);
    free(prm->c2p_index);
    free(prm->c2p_coe);
    free(prm);
    return;
}
void trans_c2p_mat(Matrix p_matrix, const Matrix c_matrix, const prm_info prm)
{
    bas_info bas = prm->bas;
    int prm_msize = prm->msize;
    int msize = bas->msize;
    int nprm = prm->nprm;
    int nbas = bas->nbas;
    Matrix tmp_matrix = malloc_matrix(prm_msize, msize);
    memset(p_matrix[0], 0, prm_msize * prm_msize * sizeof(double));
    for (int i = 0; i < prm->c2p_len; i++)
    {
        int c_id = prm->c2p_index[i * 2 + 0];
        int p_id = prm->c2p_index[i * 2 + 1];
        double c2p_d = prm->c2p_coe[i];
        int d_c = xint_gtolen(bas(ANGULAR_VAL, c_id));
        int p_c = bas->shls_p[c_id];
        int d_p = xint_gtolen(prm(ANGULAR_VAL, p_id));
        int p_p = prm->shls_p[p_id];
        for (int u = 0; u < d_c; u++)
            for (int v = 0; v < d_p; v++)
                for (int j = 0; j < msize; j++)
                    tmp_matrix[p_p + v][j] += c2p_d * c_matrix[p_c + u][j];
    }
    for (int i = 0; i < prm->c2p_len; i++)
    {
        int c_id = prm->c2p_index[i * 2 + 0];
        int p_id = prm->c2p_index[i * 2 + 1];
        double c2p_d = prm->c2p_coe[i];
        int d_c = xint_gtolen(bas(ANGULAR_VAL, c_id));
        int p_c = bas->shls_p[c_id];
        int d_p = xint_gtolen(prm(ANGULAR_VAL, p_id));
        int p_p = prm->shls_p[p_id];
        for (int u = 0; u < d_c; u++)
            for (int v = 0; v < d_p; v++)
                for (int j = 0; j < msize; j++)
                    p_matrix[p_p + v][j] += c2p_d * tmp_matrix[j][p_c + u];
    }
    free_matrix(tmp_matrix);
    return;
}

void trans_p2c_mat(Matrix c_matrix, const Matrix p_matrix, const prm_info prm)
{
    bas_info bas = prm->bas;
    int prm_msize = prm->msize;
    int msize = bas->msize;
    int nprm = prm->nprm;
    int nbas = bas->nbas;
    Matrix tmp_matrix = malloc_matrix(prm_msize, msize);
    memset(c_matrix[0], 0, msize * msize * sizeof(double));
    for (int i = 0; i < prm->c2p_len; i++)
    {
        int c_id = prm->c2p_index[i * 2 + 0];
        int p_id = prm->c2p_index[i * 2 + 1];
        double c2p_d = prm->c2p_coe[i];
        int d_c = xint_gtolen(bas(ANGULAR_VAL, c_id));
        int p_c = bas->shls_p[c_id];
        int d_p = xint_gtolen(prm(ANGULAR_VAL, p_id));
        int p_p = prm->shls_p[p_id];
        for (int u = 0; u < d_c; u++)
            for (int v = 0; v < d_p; v++)
                for (int j = 0; j < msize; j++)
                    tmp_matrix[p_c + u][j] += c2p_d * p_matrix[p_p + v][j];
    }
    for (int i = 0; i < prm->c2p_len; i++)
    {
        int c_id = prm->c2p_index[i * 2 + 0];
        int p_id = prm->c2p_index[i * 2 + 1];
        double c2p_d = prm->c2p_coe[i];
        int d_c = xint_gtolen(bas(ANGULAR_VAL, c_id));
        int p_c = bas->shls_p[c_id];
        int d_p = xint_gtolen(prm(ANGULAR_VAL, p_id));
        int p_p = prm->shls_p[p_id];
        for (int u = 0; u < d_c; u++)
            for (int v = 0; v < d_p; v++)
                for (int j = 0; j < msize; j++)
                    c_matrix[p_c + u][j] += c2p_d * tmp_matrix[j][p_p + v];
    }
    free_matrix(tmp_matrix);
    return;
}
inline double get_prm_exp(int shli, const prm_info prm)
{
    return prm->value[prm(EXP_IND, shli)];
}
inline double get_prm_coe(int shli, const prm_info prm)
{
    return prm->value[prm(COEFF_IND, shli)];
}

void print_mol_info_test(mol_info mol) {

  printf("natm = %d\n",mol->atm->natm);
  printf("nbas = %d\n",mol->bas->nbas);
  printf("msize = %d\n",mol->bas->msize);
  return;

}

mol_info init_mol(const char *mol_fname, const char *bas_fname,int unit_bohr)
{
//    printf("mol_fname = %s\n",mol_fname);
//    printf("bas_fname = %s\n",bas_fname);
//    printf("unit_bohr = %d\n",unit_bohr);
    mol_info mol = (mol_info)malloc(sizeof(struct MolInfo));
//    printf("OK 1\n");
    mol->atm = init_atm(mol_fname,unit_bohr);
//    printf("OK 2\n");
    mol->bas = init_bas(bas_fname, mol->atm);
//    printf("OK 3\n");
    mol->prm = init_prm(mol->bas);
//    printf("OK 4\n");
    mol->pair = init_pair(mol->bas);
//    printf("OK 5\n");
    mol->prm_pair = init_prm_pair(mol->prm);
//    printf("OK 6\n");
    return mol;
}
void del_mol(mol_info mol)
{
    del_atm(mol->atm);
    del_bas(mol->bas);
    del_prm(mol->prm);
    del_pair(mol->pair);
    del_pair(mol->prm_pair);
    free(mol);
    return;
}
void print_mol(const mol_info mol)
{
    printf("  The charge is %2d and multi spin is %2d.\n", mol->atm->charge, mol->atm->mult);
    printf("  The alpha electron's number is %5d and beta electron's number is %5d.\n", mol->atm->alpha_num, mol->atm->beta_num);
    printf("  The structure of molecule is shown as:\n");
    print_atm(mol->atm);
    printf("  The number of shells is %5d and the number of basis functions is %5d\n", get_mol_nbas(mol), get_mol_msize(mol));
    print_bas(mol->bas);
    return;
}
int get_mol_charge(const mol_info mol)
{
    return mol->atm->charge;
}
int get_mol_mult(const mol_info mol)
{
    return mol->atm->mult;
}
int get_mol_natm(const mol_info mol)
{
    return mol->atm->natm;
}
int get_mol_nbas(const mol_info mol)
{
    return mol->bas->nbas;
}
int get_mol_msize(const mol_info mol)
{
    return mol->bas->msize;
}

double cal_nuc_rep(const mol_info mol)
{
    double energy = 0.;
    const atm_info atm = mol->atm;
    int natm = get_mol_natm(mol);
    double dist;
    for (int i = 0; i < natm; i++)
    {
        const double *A = get_atm_coord(i, atm);
        for (int j = 0; j < i; j++)
        {
            const double *B = get_atm_coord(j, atm);
            dist = (A[0] - B[0]) * (A[0] - B[0]) +
                   (A[1] - B[1]) * (A[1] - B[1]) +
                   (A[2] - B[2]) * (A[2] - B[2]);
            dist = sqrt(dist);
            energy += mol->atm(ELEMENT_VAL, i) * mol->atm(ELEMENT_VAL, j) / dist;
        }
    }
    return energy;
}

extern char *get_ang_tag(int i, int ang)
{
    static char result[16];
    if (ang == 0)
        strcpy(result, "S");
    else if (ang == 1)
    {
        if (i == 0)
            strcpy(result, "PX");
        else if (i == 1)
            strcpy(result, "PY");
        else
            strcpy(result, "PZ");
    }
    else if (ang == 2)
    {
        if (i == 0)
            strcpy(result, "DXX");
        else if (i == 1)
            strcpy(result, "DXY");
        else if (i == 2)
            strcpy(result, "DXZ");
        else if (i == 3)
            strcpy(result, "DYY");
        else if (i == 4)
            strcpy(result, "DYZ");
        else
            strcpy(result, "DZZ");
    }
    else if (ang == 3)
    {
        if (i == 0)
            strcpy(result, "FXXX");
        else if (i == 1)
            strcpy(result, "FXXY");
        else if (i == 2)
            strcpy(result, "FXXZ");
        else if (i == 3)
            strcpy(result, "FXYY");
        else if (i == 4)
            strcpy(result, "FXYZ");
        else if (i == 5)
            strcpy(result, "FXZZ");
        else if (i == 6)
            strcpy(result, "FYYY");
        else if (i == 7)
            strcpy(result, "FYYZ");
        else if (i == 8)
            strcpy(result, "FYZZ");
        else
            strcpy(result, "FZZZ");
    }
    else if (ang == 4)
    {
        if (i == 0)
            strcpy(result, "GXXXX");
        else if (i == 1)
            strcpy(result, "GXXXY");
        else if (i == 2)
            strcpy(result, "GXXXZ");
        else if (i == 3)
            strcpy(result, "GXXYY");
        else if (i == 4)
            strcpy(result, "GXXYZ");
        else if (i == 5)
            strcpy(result, "GXXZZ");
        else if (i == 6)
            strcpy(result, "GXYYY");
        else if (i == 7)
            strcpy(result, "GXYYZ");
        else if (i == 8)
            strcpy(result, "GXYZZ");
        else if (i == 9)
            strcpy(result, "GXZZZ");
        else if (i == 10)
            strcpy(result, "GYYYY");
        else if (i == 11)
            strcpy(result, "GYYYZ");
        else if (i == 12)
            strcpy(result, "GYYZZ");
        else if (i == 13)
            strcpy(result, "GYZZZ");
        else if (i == 14)
            strcpy(result, "GZZZZ");
    }
    else if (ang == 5)
    {
        if (i == 0)
            strcpy(result, "HXXXXX");
        else if (i == 1)
            strcpy(result, "HXXXXY");
        else if (i == 2)
            strcpy(result, "HXXXXZ");
        else if (i == 3)
            strcpy(result, "HXXXYY");
        else if (i == 4)
            strcpy(result, "HXXXYZ");
        else if (i == 5)
            strcpy(result, "HXXXZZ");
        else if (i == 6)
            strcpy(result, "HXXYYY");
        else if (i == 7)
            strcpy(result, "HXXYYZ");
        else if (i == 8)
            strcpy(result, "HXXYZZ");
        else if (i == 9)
            strcpy(result, "HXXZZZ");
        else if (i == 10)
            strcpy(result, "HXYYYY");
        else if (i == 11)
            strcpy(result, "HXYYYZ");
        else if (i == 12)
            strcpy(result, "HXYYZZ");
        else if (i == 13)
            strcpy(result, "HXYZZZ");
        else if (i == 14)
            strcpy(result, "HXZZZZ");
        else if (i == 15)
            strcpy(result, "HYYYYY");
        else if (i == 16)
            strcpy(result, "HYYYYZ");
        else if (i == 17)
            strcpy(result, "HYYYZZ");
        else if (i == 18)
            strcpy(result, "HYYZZZ");
        else if (i == 19)
            strcpy(result, "HYZZZZ");
        else if (i == 20)
            strcpy(result, "HZZZZZ");
    }
    return result;
}

void print_cmatrix(Matrix matrix, int msize, const mol_info mol, const Vector egn_value, int orb_num)
{
    char *ang_tag, *ele_tag;
    int orb_batch = ceil(msize / 5.);
    for (int i = 0, u = 0; i < orb_batch; i++)
    {
        int v = u;
        printf("TAG:                   ");
        for (int k = 0; k < 5; k++, v++)
        {
            if (v >= msize)
                break;
            printf("  %-5d      ", v);
        }
        printf("\n");
        v = u;
        printf("ENG:                ");
        for (int k = 0; k < 5; k++, v++)
        {
            if (v >= msize)
                break;
            printf("%12.6f ", egn_value[v]);
        }
        printf("\n");
        printf("================================================="
               "=====================================\n");
        for (int j = 0, tag = 0; j < get_mol_nbas(mol); j++)
        {
            int di;
            di = xint_gtolen(mol->bas(ANGULAR_VAL, j));
            for (int m = 0; m < di; m++, tag++)
            {
                v = u;
                ang_tag = get_ang_tag(m, mol->bas(ANGULAR_VAL, j));
                ele_tag = charge2ele(mol->atm(ELEMENT_VAL, mol->bas(ATOM_IND, j)));
                printf("%-5d %-2s %-4d %-6s", tag, ele_tag, mol->bas(ATOM_IND, j), ang_tag);
                for (int k = 0; k < 5; k++, v++)
                {
                    if (v >= msize)
                        continue;
                    printf("%12.6f ", matrix[i * 5 + k][mol->bas->shls_p[j] + m]);
                }
                printf("\n");
                free(ele_tag);
                ele_tag = NULL;
            }
        }
        printf("================================================="
               "=====================================\n");
        u += 5;
    }
    fflush(stdout);
    return;
}

mol_info get_mol_seg(int charge, int mult, int begin_id, int end_id, const mol_info mol)
{
    mol_info seg = (mol_info)malloc(sizeof(struct MolInfo));
    seg->atm = get_atm_seg(charge, mult, begin_id, end_id, mol->atm);
    seg->bas = init_bas(mol->bas->bas_fname, seg->atm);
    seg->pair = init_pair(seg->bas);
    seg->prm = init_prm(seg->bas);
    seg->prm_pair = init_prm_pair(seg->prm);
    mol->atm->mol_fname = NULL;
    return seg;
}
