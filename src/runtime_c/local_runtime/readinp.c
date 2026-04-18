#define check_split xmvb_cpp_check_split
#define parse_blw_params xmvb_cpp_parse_blw_params
#define parse_eda_params xmvb_cpp_parse_eda_params
#define parse_sav_info xmvb_cpp_parse_sav_info
#define parse_frozen_vb_orbital xmvb_cpp_parse_frozen_vb_orbital
#define get_string_head xmvb_cpp_get_string_head
#define lower_case xmvb_cpp_lower_case
#define upper_case xmvb_cpp_upper_case
#define getcomputemethod xmvb_cpp_getcomputemethod
#define parse_dft_param xmvb_cpp_parse_dft_param
#define getcom xmvb_cpp_getcom
#define parse_pople_basis xmvb_cpp_parse_pople_basis
#define getbasisaux xmvb_cpp_getbasisaux
#define init_inp_param xmvb_cpp_init_inp_param
#define readinp xmvb_cpp_readinp
#define readinp_py xmvb_cpp_readinp_py
#define get_inttyp_inp_str xmvb_cpp_get_inttyp_inp_str

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <malloc.h>
#include <unistd.h>
#include <assert.h>
#include <limits.h>
#include "inpout/input.h"
#include "scf/hf.h"
#include "xc_funcs.h"
#include "vb/vb.h"
#include "gopt/gopt.h"

static int should_log_readinp_debug(void);
static int should_echo_input_file(void);

static const char* get_runtime_temp_directory(void) {
  const char* temp_directory = getenv("TMPDIR");
  if (temp_directory == NULL || temp_directory[0] == '\0') {
    temp_directory = "/tmp";
  }
  return temp_directory;
}

static void build_runtime_temp_path(
    char* output_path,
    size_t output_capacity,
    const char* prefix,
    const char* suffix) {
  const int written = snprintf(
      output_path,
      output_capacity,
      "%s/%s-%ld%s",
      get_runtime_temp_directory(),
      prefix,
      (long)getpid(),
      suffix);
  if (written < 0 || (size_t)written >= output_capacity) {
    fprintf(stderr, "Error! runtime temporary file path is too long.\n");
    exit(1);
  }
}

static FILE* open_required_file(
    const char* path,
    const char* mode,
    const char* description) {
  if (should_log_readinp_debug()) {
    fprintf(stderr, "readinp debug: fopen(%s, %s)\n", path, mode);
    fflush(stderr);
  }
  FILE* file = fopen(path, mode);
  if (file == NULL) {
    fprintf(stderr, "Error! failed to open %s: %s\n", description, path);
    perror("fopen");
    exit(1);
  }
  return file;
}

static void trim_trailing_newline(char* line) {
  const size_t line_length = strlen(line);
  if (line_length > 0 && line[line_length - 1] == '\n') {
    line[line_length - 1] = '\0';
  }
}

static void read_required_line(
    FILE* input_file,
    char* line_buffer,
    size_t line_buffer_capacity,
    const char* description) {
  if (fgets(line_buffer, (int)line_buffer_capacity, input_file) == NULL) {
    fprintf(stderr, "Error! failed to read %s.\n", description);
    exit(1);
  }
  trim_trailing_newline(line_buffer);
}

static void append_string_or_die(char* destination, size_t capacity, const char* source) {
  const size_t destination_length = strlen(destination);
  const size_t source_length = strlen(source);
  if (destination_length + source_length >= capacity) {
    fprintf(stderr, "Error! string append would overflow the destination buffer.\n");
    exit(1);
  }
  memcpy(destination + destination_length, source, source_length + 1);
}

static void append_string_n_or_die(
    char* destination,
    size_t capacity,
    const char* source,
    size_t source_length) {
  const size_t destination_length = strlen(destination);
  if (destination_length + source_length >= capacity) {
    fprintf(stderr, "Error! string append would overflow the destination buffer.\n");
    exit(1);
  }
  memcpy(destination + destination_length, source, source_length);
  destination[destination_length + source_length] = '\0';
}

static void build_prefixed_path_or_die(
    char* destination,
    size_t capacity,
    const char* prefix,
    const char* leaf_name) {
  const int written = snprintf(destination, capacity, "%s%s", prefix, leaf_name);
  if (written < 0 || (size_t)written >= capacity) {
    fprintf(stderr, "Error! path is too long.\n");
    exit(1);
  }
}

static int should_log_readinp_debug(void) {
  const char* debug_flag = getenv("XMVB_CPP_DEBUG_READINP");
  return debug_flag != NULL && debug_flag[0] != '\0' && strcmp(debug_flag, "0") != 0;
}

static int should_echo_input_file(void) {
  const char* echo_flag = getenv("XMVB_CPP_ECHO_INPUT");
  if (echo_flag == NULL) {
    return 1;
  }
  if (echo_flag[0] == '\0' ||
      strcmp(echo_flag, "0") == 0 ||
      strcmp(echo_flag, "false") == 0 ||
      strcmp(echo_flag, "FALSE") == 0 ||
      strcmp(echo_flag, "no") == 0 ||
      strcmp(echo_flag, "NO") == 0) {
    return 0;
  }
  return 1;
}

void check_split(char **split,int nlen,int *nsplit) {
  int n=*nsplit;
  int i=0;
  while (i<n-1) {
    int ilen=strlen(split[i]);
    if (split[i][ilen-1]=='=' || split[i][ilen-1]==',') {
      strncpy(split[i]+ilen,split[i+1],strlen(split[i+1]));
      for (int j=i+1;j<n-1;j++)
        strncpy(split[j],split[j+1],nlen-1);
      n--;
    }
    i++;
  }

  i=1;
  while (i<n) {
    if (split[i][0]=='=' || split[i][0]==',') {
      int ilen=strlen(split[i]);
      int ilen0=strlen(split[i-1]);
      strncpy(split[i-1]+ilen0,split[i],ilen);
      for (int j=i;j<n;j++)
        strncpy(split[j],split[j+1],nlen-1);
      n--;
    }
    else {
      i++;
    }
  }

  *nsplit=n;

  return;
}


int parse_blw_params(inp_info inp_str,FILE *fp) {
  char cline[1024];

  memset(cline,0,sizeof(cline));

  int len,len1,len2,nsplit;
  char *ip=NULL;
  char keyword[100];
  int itemp;

  char** split=malloc_string_array(1024,1024);

  memset(keyword,0,sizeof(keyword));
  while(fgets(cline,1024,fp)!=NULL) {
    if (strstr(cline,"$END")!=NULL)
      break;
    len=strlen(cline);
    ip=strstr(cline,"=");
    if (ip!=NULL) {
      len2=strlen(ip);
      len1=len-len2;
      strncpy(keyword,cline,len1);
      if (strstr(keyword,"NBLK")!=NULL) {
        itemp=(int)atoi(ip+1);
        if (inp_str->nblw>0 && inp_str->nblw!=itemp) {
          printf("Error in reading NMOL in $BLW. Got NBLK=%d from previous input but now with NBLK=%d\n",inp_str->nblw,itemp);
          exit(1);
        }
        else if (inp_str->nblw==0)
          inp_str->nblw=itemp;
      }
      else {
        nsplit=0;
        split_string(ip+1,split,1024,&nsplit);
        if (inp_str->nblw>0 && inp_str->nblw!=nsplit) {
          printf("Error in reading monomers in $BLW. Got NBLK=%d from previous input but now with NBLK=%d from keyword %s\n",inp_str->nblw,nsplit,keyword);
          exit(1);
        }
        if (inp_str->nblw==0)
          inp_str->nblw=nsplit;
        if (inp_str->monomers==NULL) {
          inp_str->monomers=(int*)malloc(sizeof(int)*inp_str->nblw*3);
          memset(inp_str->monomers,0,sizeof(int)*inp_str->nblw*3);
        }
        if (strstr(keyword,"MATOM")!=NULL)
          for (int i=0;i<inp_str->nblw;i++)
            *(inp_str->monomers+i)=(int)atoi(split[i]);
        else if (strstr(keyword,"MCHARGE")!=NULL)
          for (int i=0;i<inp_str->nblw;i++)
            *(inp_str->monomers+i+inp_str->nblw)=(int)atoi(split[i]);
        else if (strstr(keyword,"MMULT")!=NULL)
          for (int i=0;i<inp_str->nblw;i++)
            *(inp_str->monomers+i+2*inp_str->nblw)=(int)atoi(split[i]);
      }
    }
    memset(cline,0,sizeof(char)*1024);
  }

  free_string_array(split);

  int tot_mul=0,sign,imul;
  int tot_charge=0;

  for (int i=0;i<inp_str->nblw;i++) {
    imul=*(inp_str->monomers+i+2*inp_str->nblw);
    sign=(imul>0?1:-1);
    imul=(sign*imul-1)*sign;
    tot_mul+=imul;
    tot_charge+=*(inp_str->monomers+i+inp_str->nblw);
  }

  if (tot_mul!=(inp_str->nmul-1)) {
    printf("Error in multiplicity in $BLW. The total multiplicity should be %d while %d is obtained\n",inp_str->nmul,tot_mul+1);
    exit(1);
  }

  if (tot_charge!=inp_str->ncharge) {
    printf("Error in charge in $BLW. The total charge should be %d while %d is obtained\n",inp_str->ncharge,tot_charge);
    exit(1);
  }

  return 0;
}

int parse_eda_params(inp_info inp_str,FILE *fp) {
  char cline[1024];

  memset(cline,0,sizeof(cline));

  int len,len1,len2,nsplit;
  char *ip=NULL;
  char keyword[100];
  int itemp;

  char** split=malloc_string_array(1024,1024);

  memset(keyword,0,sizeof(keyword));
  while(fgets(cline,1024,fp)!=NULL) {
    if (strstr(cline,"$END")!=NULL)
      break;
    len=strlen(cline);
    ip=strstr(cline,"=");
    if (ip!=NULL) {
      len2=strlen(ip);
      len1=len-len2;
      strncpy(keyword,cline,len1);
      if (strstr(keyword,"NMOL")!=NULL) {
        itemp=(int)atoi(ip+1);
        if (inp_str->neda>0 && inp_str->neda!=itemp) {
          printf("Error in reading NMOL in $EDA. Got NMOL=%d from previous input but now with NMOL=%d\n",inp_str->neda,itemp);
          exit(1);
        }
        else if (inp_str->neda==0)
          inp_str->neda=itemp;
      }
      else {
        nsplit=0;
        split_string(ip+1,split,1024,&nsplit);
        if (inp_str->neda>0 && inp_str->neda!=nsplit) {
          printf("Error in reading monomers in $EDA. Got NMOL=%d from previous input but now with NMOL=%d from keyword %s\n",inp_str->neda,nsplit,keyword);
          exit(1);
        }
        if (inp_str->neda==0)
          inp_str->neda=nsplit;
        if (inp_str->monomers==NULL) {
          inp_str->monomers=(int*)malloc(sizeof(int)*inp_str->neda*3);
          memset(inp_str->monomers,0,sizeof(int)*inp_str->neda*3);
        }
        if (strstr(keyword,"MATOM")!=NULL)
          for (int i=0;i<inp_str->neda;i++)
            *(inp_str->monomers+i)=(int)atoi(split[i]);
        else if (strstr(keyword,"MCHARGE")!=NULL)
          for (int i=0;i<inp_str->neda;i++)
            *(inp_str->monomers+i+inp_str->neda)=(int)atoi(split[i]);
        else if (strstr(keyword,"MMULT")!=NULL)
          for (int i=0;i<inp_str->neda;i++)
            *(inp_str->monomers+i+2*inp_str->neda)=(int)atoi(split[i]);
      }
    }
    memset(cline,0,sizeof(char)*1024);
  }

  free_string_array(split);

  int tot_mul=0,sign,imul;
  int tot_charge=0;

  for (int i=0;i<inp_str->neda;i++) {
    imul=*(inp_str->monomers+i+2*inp_str->neda);
    sign=(imul>0?1:-1);
    imul=(sign*imul-1)*sign;
    tot_mul+=imul;
    tot_charge+=*(inp_str->monomers+i+inp_str->neda);
  }

  if (tot_mul!=(inp_str->nmul-1)) {
    printf("Error in multiplicity in $EDA. The total multiplicity should be %d while %d is obtained\n",inp_str->nmul,tot_mul+1);
    exit(1);
  }

  if (tot_charge!=inp_str->ncharge) {
    printf("Error in charge in $EDA. The total charge should be %d while %d is obtained\n",inp_str->ncharge,tot_charge);
    exit(1);
  }

  return 0;
}


// Get the information for state-average calculation
int parse_sav_info(char *kwd, char *kval, int *idxstate, double *wstate, int *nsav) {
    char *ip=strchr(kwd,'(');
    char *is=strchr(kwd,')');
    int plen=strlen(ip);
    int slen=strlen(is);
    int nlen=plen-slen;
    char *start_state=(char*)malloc(sizeof(char)*nlen);
    memset(start_state, '\0', sizeof(char)*nlen);
    strncpy(start_state,ip+1,nlen-1);
    int start=atoi(start_state);
    int nsplit=0;

    char** split=malloc_string_array(MAX_STATE,1024);

    start--;
    if (start<0) {
      printf("Error! The index in %s is not valid\n",kwd);
      exit(1);
    }

    for (int i=0;i<strlen(kval);i++)
        if (*(kval+i)==',')
            *(kval+i)=' ';

    split_string(kval,split,1024,&nsplit);

    for (int i=0;i<nsplit;i++) {
        *(wstate+(*nsav))=(double)atof(split[i]);
        *(idxstate+(*nsav))=start+i;
        (*nsav)++;
    }

    free(start_state);

    free_string_array(split);

    return 0;
}


// Get the information for state-average calculation
List parse_frozen_vb_orbital(char *kval, int *nfroz) {

    List orb_tmp = malloc_list(200); 
    int nsplit = 0; 

    char** split=malloc_string_array(MAX_STATE,1024);

    for (int i = 0; i < strlen(kval); i++) {
        if (*(kval + i) == ',') {
            *(kval + i) = ' '; 
        }
    }

    split_string(kval, split, 1024, &nsplit); 
    *nfroz = 0; 
    expand_str_orb(split, orb_tmp, nsplit, nfroz); 

    List froz_list = malloc_list(*nfroz); 

    for (int i = 0; i < *nfroz; i++) {
        froz_list[i] = orb_tmp[i]; 
    }

    free_list(orb_tmp); 

    free_string_array(split);

    return froz_list; 
}


// Return the position of the first character in str that is not blank 
char* get_string_head(char *str) {

    char *head=NULL;

    for (int i=0;i<strlen(str);i++) {
        if (*(str+i)!=' ') {
            head=str+i;
            break;
        }
    }

    return head;
}

// Change the upper case character in string to lower case
void lower_case(char *string) {
    int i;
    i=0;
    while(*(string+i)!='\0') {
        if (*(string+i) >= 'A' && *(string+i) <= 'Z')
            *(string+i)+=32;
        i++;
    }
}

// Change the lower case character in string to upper case
void upper_case(char *string) {
    int i;
    i=0;
    while(*(string+i)!='\0') {
        if (*(string+i) >= 'a' && *(string+i) <= 'z')
            *(string+i)-=32;
        i++;
    }
}

// @brief Anaylize the keword "METHOD" and get the specified method 
// 
// @param[in] str The specified method. 
// @param[out] dftfunc The name of functional used if doing DFT calculation. 
// @param[out] ihf_type Type of HF calculation to be done. 
// @param[out] dodft 1 if doing DFT, 0 if not. 
// @param[out] dovbscf 1 if doing VBSCF, 0 if not. 
// @param[out] dobovb 1 if doing BOVB, 0 if not. 
// @param[out] dovbcis 1 if doing VBCIS, 0 if not. 
// @param[out] dovbcisd 1 if doing VBCISD, 0 if not. 
// @param[out] dovbcids 1 if doing VBCIDS, 0 if not. 
// @param[out] dovbpt2 1 if doing VBPT2, 0 if not. 
// @param[out] boysloc 1 if doing Boys localization, 0 if not. 
// 
// @author 
// @date 
void getcomputemethod(char *str,char *dftfunc,int *ihf_type,int *dodft,int *dovbscf,int *dobovb,int *dovbcis,int *dovbcisd,int *dovbcids,int *dovbpt2,int *boysloc, int *pmloc) {

#define  nfunc 8
  int i,j;
  char funcname[nfunc][20]={"BLYP","B3LYP-D3BJ","B3LYP-D3","B3LYP","BHHLYP","PW91","PBE0","PBE"};
  char rotyp_str[4][2]={"RO","CU","R","U"};
  int  rotyp_int[4]={ROHF_WORK,CUHF_WORK,RHF_WORK,UHF_WORK};
  char dft[40];
  int localize=(*boysloc)+(*pmloc);

  if (strstr(str,"RHF")!=NULL) {
    *ihf_type=RHF_WORK;
    *dovbscf = 0; 
  }
  else if (strstr(str,"UHF")!=NULL) {
    *ihf_type=UHF_WORK;
    *dovbscf = 0; 
  }
  else if (strstr(str,"ROHF")!=NULL) {
    *ihf_type=ROHF_WORK;
    *dovbscf = 0; 
  }
  else if (strstr(str,"CUHF")!=NULL) {
    *ihf_type=CUHF_WORK;
    *dovbscf = 0; 
  }
  else if (strstr(str,"HF")!=NULL) {
    *ihf_type=-2;
    *dovbscf = 0; 
  }
  else if (strstr(str,"VBSCF")!=NULL)
    *dovbscf=1;
  else if (strstr(str,"BOVB")!=NULL)
    *dobovb=1;
  else if (strstr(str,"VBCISD")!=NULL) {
    *dovbscf=1;
    if (localize==0)
      *boysloc=1;
    *dovbcisd=1;
  }
  else if (strstr(str,"VBCIDS")!=NULL) {
    *dovbscf=1;
    if (localize==0)
      *boysloc=1;
    *dovbcids=1;
  }
  else if (strstr(str,"VBCIS")!=NULL) {
    *dovbscf=1;
    if (localize==0)
      *boysloc=1;
    *dovbcis=1;
  }
  else if (strstr(str,"VBPT2")!=NULL) {
    *dovbscf=1;
    *dovbpt2=1;
  }
  else {
    for (i=0;i<nfunc;i++)
      if (strstr(str,funcname[i])!=NULL) {
        *ihf_type=-2;
        strncpy(dftfunc,funcname[i],strlen(funcname[i]));
        *dodft=1;
        *dovbscf = 0;
        for (j=0;j<4;j++) {
          memset(dft,0,sizeof(dft));
          append_string_or_die(dft, sizeof(dft), rotyp_str[j]);
          append_string_or_die(dft, sizeof(dft), funcname[i]);
          if (strstr(str,dft)!=NULL) {
              *ihf_type=rotyp_int[j];
              break;
          }
        }
      }
  }

  return;
}

// @brief Parse parameters for DFT calculation 
//
// @param[out] inp_str->ndft Number of functional used for Libcint. 
// @param[out] inp_str->hf_frac HF exchange fraction used in DFT. 
// @param[out] inp_str->dft_id Functional ID for Libcint. 
// @param[out] inp_str->dft_frac DFT functional fraction used in DFT. 
// @param[out] inp_str->grid_type Grid type used in DFT. 
//
// @author 
// @date 
int parse_dft_param(inp_info inp_str) {

//  printf("dftfunc = %s\n",inp_str->dftfunc);

  if (strstr(inp_str->dftfunc,"BLYP")!=NULL) {
    inp_str->ndft=2;
    inp_str->hf_frac=0.0;
    inp_str->dft_id=(int *)malloc(inp_str->ndft*sizeof(int));
    inp_str->dft_frac=(double*)malloc(inp_str->ndft*sizeof(double));
    *inp_str->dft_id=XC_GGA_X_B88;
    *(inp_str->dft_id+1)=XC_GGA_C_LYP;
    *inp_str->dft_frac=1.;
    *(inp_str->dft_frac+1)=1.;
    inp_str->grid_type=FINE_GRIDS;
    inp_str->dft_name=-1;
    inp_str->disp_type=false;
  }
  else if (strstr(inp_str->dftfunc,"B3LYP-D3BJ")!=NULL) {
//    printf("setup for B3LYP-D3BJ\n");
    inp_str->ndft=1;
    inp_str->hf_frac=0.2;
    inp_str->dft_id=(int *)malloc(inp_str->ndft*sizeof(int));
    inp_str->dft_frac=(double*)malloc(inp_str->ndft*sizeof(double));
    *inp_str->dft_id=XC_HYB_GGA_XC_B3LYP;
    *inp_str->dft_frac=1.;
    inp_str->grid_type=MEDIUM_GRIDS;
    inp_str->dft_name=B3LYP;
    inp_str->disp_type=DFT_D3_BJ;
  }
  else if (strstr(inp_str->dftfunc,"B3LYP-D3")!=NULL) {
//    printf("setup for B3LYP-D3\n");
    inp_str->ndft=1;
    inp_str->hf_frac=0.2;
    inp_str->dft_id=(int *)malloc(inp_str->ndft*sizeof(int));
    inp_str->dft_frac=(double*)malloc(inp_str->ndft*sizeof(double));
    *inp_str->dft_id=XC_HYB_GGA_XC_B3LYP;
    *inp_str->dft_frac=1.;
    inp_str->grid_type=MEDIUM_GRIDS;
    inp_str->dft_name=B3LYP;
    inp_str->disp_type=DFT_D3;
  }
  else if (strstr(inp_str->dftfunc,"B3LYP")!=NULL) {
//    printf("setup for B3LYP\n");
    inp_str->ndft=1;
    inp_str->hf_frac=0.2;
    inp_str->dft_id=(int *)malloc(inp_str->ndft*sizeof(int));
    inp_str->dft_frac=(double*)malloc(inp_str->ndft*sizeof(double));
    *inp_str->dft_id=XC_HYB_GGA_XC_B3LYP;
    *inp_str->dft_frac=1.;
    inp_str->grid_type=MEDIUM_GRIDS;
    inp_str->dft_name=B3LYP;
    inp_str->disp_type=false;
  }
  else if (strstr(inp_str->dftfunc,"BHHLYP")!=NULL) {
    inp_str->ndft=2;
    inp_str->hf_frac=0.5;
    inp_str->dft_id=(int *)malloc(inp_str->ndft*sizeof(int));
    inp_str->dft_frac=(double*)malloc(inp_str->ndft*sizeof(double));
    *inp_str->dft_id=XC_GGA_X_B88;
    *(inp_str->dft_id+1)=XC_GGA_C_LYP;
    *inp_str->dft_frac=0.5;
    *(inp_str->dft_frac+1)=1.;
    inp_str->grid_type=FINE_GRIDS;
    inp_str->dft_name=-1;
    inp_str->disp_type=false;
  }
  else if (strstr(inp_str->dftfunc,"PW91")!=NULL) {
    inp_str->ndft=2;
    inp_str->hf_frac=0.0;
    inp_str->dft_id=(int *)malloc(inp_str->ndft*sizeof(int));
    inp_str->dft_frac=(double*)malloc(inp_str->ndft*sizeof(double));
    *inp_str->dft_id=XC_GGA_X_PW91;
    *(inp_str->dft_id+1)=XC_GGA_C_PW91;
    *inp_str->dft_frac=1.;
    *(inp_str->dft_frac+1)=1.;
    inp_str->grid_type=FINE_GRIDS;
    inp_str->dft_name=-1;
    inp_str->disp_type=false;
  }
  else if (strstr(inp_str->dftfunc,"PBE0")!=NULL) {
    inp_str->ndft=1;
    inp_str->hf_frac=0.25;
    inp_str->dft_id=(int *)malloc(inp_str->ndft*sizeof(int));
    inp_str->dft_frac=(double*)malloc(inp_str->ndft*sizeof(double));
    *inp_str->dft_id=XC_HYB_GGA_XC_PBEH;
    *inp_str->dft_frac=1.;
    inp_str->grid_type=MEDIUM_GRIDS;
    inp_str->dft_name=-1;
    inp_str->disp_type=false;
  }
  else if (strstr(inp_str->dftfunc,"PBE")!=NULL) {
    inp_str->ndft=2;
    inp_str->hf_frac=0.0;
    inp_str->dft_id=(int *)malloc(inp_str->ndft*sizeof(int));
    inp_str->dft_frac=(double*)malloc(inp_str->ndft*sizeof(double));
    *inp_str->dft_id=XC_GGA_X_PBE;
    *(inp_str->dft_id+1)=XC_GGA_C_PBE;
    *inp_str->dft_frac=1.;
    *(inp_str->dft_frac+1)=1.;
    inp_str->grid_type=FINE_GRIDS;
    inp_str->dft_name=-1;
    inp_str->disp_type=false;
  }

    return 0;
}

// Analyze keywords in $CTR section 
int getcom(inp_info inp_str, FILE *fp) {

    char cline[1024];
    int nsplit;
//    char split[1024][1024];
    char *eq;
    char kwd[40];
    char *kval;
    int ilen;

    char** split=malloc_string_array(1024,1024);

    // Get all the keywords and store in split, and store the number of keywords in nsplit 
    nsplit=0;
    while(fgets(cline,1024,fp)!=NULL) {
        *(cline+strcspn(cline,"\r\n"))=0;

        if (strstr(cline,"$END")!=NULL)
            break;
        split_string(cline,split,1024,&nsplit);
    }

    check_split(split,1024,&nsplit);

    // Analyze keywords 
    for (int i = 0; i < nsplit; i++) {
        eq = strchr(split[i],'=');
        if (eq != NULL) { // Keywords with value
            memset(kwd,0,sizeof(kwd));
            strncpy(kwd,split[i],strlen(split[i])-strlen(eq)); // Get the keyword before "=" and store in kwd
            kval=eq+1; // Get the value after "=" and store in kval

            if (strstr(kwd,"METHOD")!=NULL) {
                getcomputemethod(kval,inp_str->dftfunc,&inp_str->ihf_type,&inp_str->dodft,&inp_str->dovbscf,&inp_str->dobovb,&inp_str->dovbcis,&inp_str->dovbcisd,&inp_str->dovbcids,&inp_str->dovbpt2,&inp_str->boysloc,&inp_str->pmloc);
            }
            else if (strstr(kwd,"CTOL")!=NULL) {
              inp_str->ctol=fabs(atof(kval));
            }
            else if (strstr(kwd,"GROUP")!=NULL) {
                inp_str->fixc=1;
                strncpy(inp_str->grpval,kval,strlen(kval));
            }
            else if (strstr(kwd,"WSTATE")!=NULL) {
//              inp_str->dosav=1;
                parse_sav_info(kwd,kval,inp_str->idxstate,inp_str->wstate,&inp_str->nsav);
            }
            else if (strstr(kwd,"NSTATE")!=NULL) {
                inp_str->iroot=atoi(kval);
            }
            else if (strstr(kwd,"OUTPUT")!=NULL) {
                if (strstr(kval,"AIM")!=NULL) {
                    inp_str->dowfn=1;
                }
            }
            else if (strstr(kwd,"IPRINT") != NULL) {
                inp_str->print_level = atoi(kval); 
            }
            else if (strstr(kwd,"CICUT")!=NULL) {
                inp_str->cicut=atoi(kval);
            }
            else if (strstr(kwd,"NCOR")!=NULL) {
                inp_str->ncor=atoi(kval);
            }
            else if (strstr(kwd,"ITMAX")!=NULL) {
                inp_str->itmax=atoi(kval);
            }
            else if (strstr(kwd,"MAX_ITER")!=NULL) {
                inp_str->itmax=atoi(kval);
            }
            else if (strstr(kwd,"NSTR")!=NULL) {
                inp_str->nstr=atoi(kval);
            }
            else if (strstr(kwd,"STR")!=NULL) {
                strncpy(inp_str->strclass,kval,strlen(kval));
                inp_str->genstr=1;
            }
            else if (strstr(kwd,"NAO")!=NULL) {
                inp_str->nao=atoi(kval);
            }
            else if (strstr(kwd,"NAE")!=NULL) {
                inp_str->nae=atoi(kval);
            }
            else if (strstr(kwd,"ISCF")!=NULL) {
                inp_str->iscf=atoi(kval);
            }
            else if (strstr(kwd,"BASIS")!=NULL) {
                strncpy(inp_str->basis_name,kval,strlen(kval));
            }
            else if (strstr(kwd,"NMUL")!=NULL) {
                inp_str->nmul=atoi(kval);
            }
            else if (strstr(kwd,"NCHARGE")!=NULL) {
                inp_str->ncharge=atoi(kval);
            }
            else if (strstr(kwd,"UNIT")!=NULL) {
                if (strstr(kval,"ANGS")!=NULL)
                    inp_str->unit_bohr=0;
                else if (strstr(kval,"BOHR")!=NULL)
                    inp_str->unit_bohr=1;
            }
            else if (strstr(kwd,"GUESS")!=NULL) {
                if (strstr(kval,"AUTO")!=NULL)
                    inp_str->iguess=GUS_AUTO;
                else if (strstr(kval,"UNIT")!=NULL)
                    inp_str->iguess=GUS_UNIT;
                else if (strstr(kval,"READ")!=NULL)
                    inp_str->iguess=GUS_READ;
                else if (strstr(kval,"RDCI")!=NULL)
                    inp_str->iguess=GUS_RDCI;
                else if (strstr(kval,"MO")!=NULL)
                    inp_str->iguess=GUS_MO;
                else if (strstr(kval,"NBO")!=NULL)
                    inp_str->iguess=GUS_NBO;
            }
            else if (strstr(kwd,"ORBTYP")!=NULL) {
                if (strstr(kval,"GEN")!=NULL)
                    inp_str->orbtyp=GEN_TYP;
                else if (strstr(kval,"HAO")!=NULL)
                    inp_str->orbtyp=HAO_TYP;
                else if (strstr(kval,"BDO")!=NULL)
                    inp_str->orbtyp=BDO_TYP;
                else if (strstr(kval,"OEO")!=NULL)
                    inp_str->orbtyp=OEO_TYP;
            }
            else if (strstr(kwd,"FRGTYP")!=NULL) {
                if (strstr(kval,"ATOM")!=NULL)
                    inp_str->frgtyp=FRG_ATM;
                else if (strstr(kval,"SAO")!=NULL)
                    inp_str->frgtyp=FRG_SAO;
            }
            else if (strstr(kwd,"WFNTYP")!=NULL) {
                if (strstr(kval,"STR")!=NULL)
                    inp_str->wfntyp=WFN_STR;
                else if (strstr(kval,"DET")!=NULL)
                    inp_str->wfntyp=WFN_DET;
            }
            else if (strstr(kwd,"VBFTYP")!=NULL) {
                if (strstr(kval,"DET")!=NULL)
                    inp_str->vbftyp=VBF_DET;
                else if (strstr(kval,"PPD")!=NULL)
                    inp_str->vbftyp=VBF_PPD;
            }
            else if (strstr(kwd,"INT")!=NULL) {
                inp_str->input_requests_ri_two_electron_mode=0;
                if (strstr(kval,"READ")!=NULL)
                    inp_str->inttyp=INT_READ;
                else if (strstr(kval,"LIBCINT")!=NULL)
                    inp_str->inttyp=INT_CINT;
                else if (strstr(kval,"XINT")!=NULL)
                    inp_str->inttyp=INT_XINT;
                else if (strstr(kval,"RI")!=NULL) {
                    inp_str->inttyp=INT_RI;
                    inp_str->input_requests_ri_two_electron_mode=1;
                }
                else if (strstr(kval,"COSX")!=NULL){
                    inp_str->inttyp=INT_COSX;
                }
                else if (strstr(kval,"FCOSX")!=NULL){
                    inp_str->inttyp=INT_FCOSX;
                }
            }
            // else if (strstr(kwd,"eri_act")!=NULL){
            //     if (strstr(kval,"COSX")!=NULL)
            //         inp_str->vberi=2;
            // }
            else if (strstr(kwd,"OPT") != NULL) 
            {
                if (strstr(kval,"CART") != NULL)
                {
                    inp_str->DoGopt = 1; 
                    inp_str->DoGradient = 1; 
                    inp_str->Gopt_Type = GOPT_CART; 
                }
                else if (strstr(kval,"ZMT") != NULL)
                {
                    inp_str->DoGopt = 1; 
                    inp_str->DoGradient = 1; 
                    inp_str->Gopt_Type = GOPT_ZMT; 
                }
                else if (strstr(kval,"TS") != NULL) {
                    inp_str->DoGopt = 1; 
                    inp_str->Gopt_Type = GOPT_CART; 
                    inp_str->DoGradient = 1; 
                    inp_str->nneg=1;
                }
//                printf("gopt type in readinp = %d %d %d\n",inp_str->Gopt_Type,GOPT_CART,GOPT_ZMT);
            }
            else if (strstr(kwd,"HESS") != NULL)
            {
                if (strstr(kval,"GUESS") != NULL)
                {
                    inp_str->Hess_Type = GHESS_GUESS; 
                }
                else if (strstr(kval,"ANALYTICAL") != NULL)
                {
                    inp_str->Hess_Type = GHESS_ANALYTICAL; 
                }
                else if (strstr(kval,"SEMINU") != NULL)
                {
                    inp_str->Hess_Type = GHESS_SEMI_NUMERICAL; 
                }
                else if (strstr(kval,"FULLNU") != NULL)
                {
                    inp_str->Hess_Type = GHESS_FULLY_NUMERICAL;
                }
            }
            else if (strstr(kwd,"MAXCYCLES") != NULL)
            {
                inp_str->Max_Gopt_Cycles = atoi(kval); 
            }
            else if (strstr(kwd,"LAM-DFVB") != NULL) 
            {
                inp_str->dovbscf = 1;
                inp_str->DoLamDFVB = 1; 
                strncpy(inp_str->DFVBfunc, kval, strlen(kval)); 
            }
            else if (strstr(kwd,"HC-DFVB") != NULL) 
            {
                inp_str->dovbscf = 1;
                inp_str->DohcDFVB = 1; 
                strncpy(inp_str->DFVBfunc, kval, strlen(kval)); 
            }
            else if (strstr(kwd,"MS-DFVB") != NULL) 
            {
                inp_str->dovbscf = 1; 
                inp_str->DoMsDFVB = 1; 
                inp_str->vbcad = 1; 
                inp_str->cad_step_size = 2.0; 
                strncpy(inp_str->DFVBfunc, kval, strlen(kval)); 
            }
            else if (strstr(kwd,"FRZORB") != NULL) 
            {
                // Usage: "FRZORB=1,3,5-7" to froze orbital 1, 3, 5, 6, 7
                inp_str->froz_list = parse_frozen_vb_orbital(kval, &inp_str->nfroz); 
            }
            else if (strstr(split[i],"VBCAD") != NULL) {
                if (strstr(kval,"C2") != NULL)
                {
                    inp_str->vbcad = 1; 
                }
                else if (strstr(kval,"CC") != NULL)
                {
                    inp_str->vbcad = 2; 
                }
                else if (strstr(kval,"RE") != NULL)
                {
                    inp_str->vbcad = 3; 
                }
                else if (strstr(kval,"LOWDIN") != NULL)
                {
                    inp_str->vbcad = 4; 
                }
                else if (strstr(kval,"INV") != NULL)
                {
                    inp_str->vbcad = 5; 
                }
                else {
                    inp_str->vbcad = atoi(kval); 
                }
            }
            else if (strstr(split[i],"CADGRID") != NULL) {
                inp_str->cad_step_size = 1.0 / atof(kval); 
            }
        }
        else { // Keywords without value
            if (strstr(split[i],"FIXC")!=NULL)
                inp_str->fixc=1;
            else if (strstr(split[i],"TBVBSCF")!=NULL) {
              inp_str->biovb=1;
            }
            else if (strstr(split[i],"READCOEF")!=NULL)
              inp_str->bio_readcoef=1;
            else if (strstr(split[i],"BOYS")!=NULL) {
                inp_str->boysloc=1;
                if (inp_str->pmloc>0)
                  inp_str->pmloc=0;
            }
            else if (strstr(split[i],"PMLOC")!=NULL) {
                inp_str->pmloc=1;
                if (inp_str->boysloc>0)
                  inp_str->boysloc=0;
            }
            else if (strstr(split[i],"DEBUG")!=NULL)
                inp_str->print_level=3;
            else if (strstr(split[i],"DIR2E")!=NULL)
                inp_str->dir2e=1;
            else if (strstr(split[i],"OPT") != NULL) {
                inp_str->DoGopt = 1; 
                inp_str->DoGradient = 1; 
            }
            else if (strstr(split[i],"GRADIENT") != NULL) {
                inp_str->DoGradient = 1; 
                inp_str->prtGradOnly = 1;
            } 
            else if (strstr(split[i],"MOLDEN") != NULL) {
                inp_str->molden = 1; 
            }
            else if (strstr(split[i],"SORT") != NULL) {
                inp_str->sort = 1; 
            }
            else if (strstr(split[i],"PUNCH") != NULL) {
              inp_str->int_punch=1;
            }
            else if (strstr(split[i],"POINTS")!=NULL)
                inp_str->read_points=1;
            else if (strstr(split[i],"VBCAD") != NULL) {
                inp_str->vbcad = 1; 
                inp_str->cad_step_size = 2.0; 
            }
            else
                getcomputemethod(split[i],inp_str->dftfunc,&inp_str->ihf_type,&inp_str->dodft,&inp_str->dovbscf,&inp_str->dobovb,&inp_str->dovbcis,&inp_str->dovbcisd,&inp_str->dovbcids,&inp_str->dovbpt2,&inp_str->boysloc,&inp_str->pmloc);
        }
    }

    if (inp_str->dodft>0) {
        parse_dft_param(inp_str);
    }
    else
        inp_str->ndft=0;

    free_string_array(split);

    return 0;
}

void parse_pople_basis(char *basis) {

  int i;
  char *iG=strchr(basis,'g');
  char basname[1024];

  if (iG==NULL) {
    printf("Error in Pople basis name %s\n",basis);
    exit(1);
  }
  
  memset(basname,0,sizeof(basname));
  int nplus=0;
  int nstar=0;
  for (i=0;i<strlen(basis);i++) {
    if (*(basis+i)=='+')
      nplus++;
    else if (*(basis+i)=='*')
      nstar++;
  }

  char *ip=strchr(iG,'(');
  char *is=strchr(iG,')');
  char *comma=strchr(iG,',');

  int nps;
  if (ip==NULL || is==NULL)
    nps=0;
  else
    nps=strlen(ip)-strlen(is);

  int np_comma,ns_comma;
  if (comma==NULL||ip==NULL)
    np_comma=0;
  else
    np_comma=strlen(ip)-strlen(comma);

  if (comma==NULL||is==NULL)
    ns_comma=0;
  else
    ns_comma=strlen(comma)-strlen(is);

  if (ip!=NULL || comma!=NULL || is!=NULL) {
    if (nps<2 || (ip==NULL && is!=NULL) || ((np_comma<2 || ns_comma<2) && comma!=NULL)) {
      printf("Error in parsing basis name %s\n",basis);
      printf("A1\n");
      exit(1);
    }
  }

  if (nplus>2 || nstar > 2) {
    printf("Basis set %s is not supported\n",basis);
    exit(1);
  }

  if (nplus == 0)
    strncpy(basname,basis,strlen(basis)-strlen(iG));
  else {
    strncpy(basname,basis,strlen(basis)-strlen(iG)-nplus);
    for (i=0;i<nplus;i++)
      strcat(basname,"p");
  }

  strcat(basname,"g");

  if (nstar == 1) 
    strcat(basname,"_d_");
  else if (nstar == 2)
    strcat(basname,"_d_p_");
  else if (ip!=NULL) {
    strcat(basname,"_");
    if (comma!=NULL) {
      append_string_n_or_die(
          basname,
          sizeof(basname),
          ip+1,
          (size_t)(strlen(ip+1)-strlen(comma)));
      strcat(basname,"_");
      append_string_n_or_die(
          basname,
          sizeof(basname),
          comma+1,
          (size_t)(strlen(comma+1)-strlen(is)));
      strcat(basname,"_");
    }
    else {
      append_string_n_or_die(
          basname,
          sizeof(basname),
          ip+1,
          (size_t)(strlen(ip+1)-strlen(is)));
      strcat(basname,"_");
    }
  }

  memset(basis,0,sizeof(char)*strlen(basis));
  strncpy(basis,basname,strlen(basname));
}

void getbasisaux(inp_info inp_str) {
  char *pople[3]={"STO-","3-21","6-31"};
  int i,ilen;
  char *basis,*aux;

  memset(inp_str->aux_name,0,sizeof(inp_str->aux_name));
  for (i=0;i<3;i++) {
    if (strstr(inp_str->basis_name,pople[i])!=NULL) {
      strcpy(inp_str->aux_name,"def2-svp-jfit.gbs");
      break;
    }
  }
  lower_case(inp_str->basis_name);
  if (strlen(inp_str->aux_name)>0) {
    parse_pople_basis(inp_str->basis_name);
  }
  else {
    strcpy(inp_str->aux_name,inp_str->basis_name);
    if (strstr(inp_str->basis_name,"cc-p")!=NULL)
      strcat(inp_str->aux_name,"-ri.gbs");
    else if (strstr(inp_str->basis_name,"def2-")!=NULL)
      strcat(inp_str->aux_name,"-jfit.gbs");
    else if (strstr(inp_str->basis_name,"ma-")!=NULL)
      strcat(inp_str->aux_name,"-jkfit.gbs");
    else {
      if (inp_str->inttyp!=INT_READ) {
        printf("Auxiliary basis set for %s is not supported\n",inp_str->basis_name);
        exit(1);
      }
    }
  }
  strcat(inp_str->basis_name,".gbs");
}

void init_inp_param(inp_info inp_str) {
  inp_str->j_grid_file=NULL;

  inp_str->biovb=0;
  inp_str->bio_readcoef=0;

  inp_str->ctol=0e0;
  inp_str->vmax=10;
  inp_str->itmax=-1;
  inp_str->dodft=0;
  inp_str->dovb=0;
  inp_str->dovbscf=1;
  inp_str->dobovb=0;
  inp_str->dovbcis=0;
  inp_str->dovbcisd=0;
  inp_str->dovbcids=0;
  inp_str->dovbpt2=0;
  inp_str->dopop=0;
  inp_str->dowfn=0;
  inp_str->doeda=0;
  inp_str->doblw=0;
  inp_str->boysloc=0;
  inp_str->pmloc=0;
  inp_str->genstr=0;
  inp_str->nstr=0;
  inp_str->nel=0;
  inp_str->nb=0;
  inp_str->nor=0;
  inp_str->nao=0;
  inp_str->nae=0;
  inp_str->iroot=0;
  inp_str->nmul=1;
  inp_str->ncharge=0;
  inp_str->unit_bohr=0;
  inp_str->ihf_type=0;
  inp_str->iguess=GUS_AUTO;
  inp_str->orbtyp=GEN_TYP;
  inp_str->frgtyp=FRG_ATM;
  inp_str->wfntyp=WFN_STR;
  inp_str->vbftyp=VBF_DET;
  inp_str->dir2e=0;
  inp_str->inttyp=0;
  inp_str->ncor=0;
  inp_str->cicut=0;
  inp_str->inci=0;
  inp_str->dobfi=0;
  inp_str->fixc=0;
//  inp_str->dosav=0;
  inp_str->nsav=0;
  inp_str->ngroup=0;
  inp_str->neda=0;
  inp_str->nblw=0;
  inp_str->monomers=NULL;
  inp_str->print_level=1;
  inp_str->molden=0; 
//  inp_str->iscf=5; 
  inp_str->iscf=0; 
  inp_str->DoLamDFVB = 0; 
  inp_str->DohcDFVB = 0; 
  inp_str->DoMsDFVB = 0; 
  inp_str->nfroz = 0; 

  inp_str->dft_id=NULL;
  inp_str->dft_frac=NULL;

  inp_str->sort = 0;

  inp_str->vbcad = 0; 
  inp_str->cad_step_size = 0.0; 

  inp_str->nao=0;
  inp_str->nae=0;

  inp_str->read_points=0;

  // Geometry optimization
  inp_str->DoGopt = GOPT_NONE; 
  inp_str->Gopt_Type = GOPT_CART; 
  inp_str->Grad_Type = GGRAD_NONE; 
  inp_str->Hess_Type = GHESS_NONE; 
  inp_str->Max_Gopt_Cycles = 0;
  inp_str->nneg=0;
  inp_str->DoGradient = 0; 
  inp_str->prtGradOnly = 0;

  inp_str->int_punch=0;
  inp_str->input_requests_ri_two_electron_mode=0;
  
  memset(inp_str->strclass,0,sizeof(inp_str->strclass));
  memset(inp_str->idxstate,0,sizeof(inp_str->idxstate));
  memset(inp_str->wstate,0,sizeof(inp_str->wstate));
  memset(inp_str->grpval,0,sizeof(inp_str->grpval));
  memset(inp_str->wfn_name,0,sizeof(inp_str->wfn_name));

  memset(inp_str->inpname,0,sizeof(inp_str->inpname));
  memset(inp_str->mol_name,0,sizeof(inp_str->mol_name));
  memset(inp_str->dftfunc,0,sizeof(inp_str->dftfunc));
  memset(inp_str->title,0,sizeof(inp_str->title));
  memset(inp_str->basis_name,0,sizeof(inp_str->basis_name));
  memset(inp_str->aux_name,0,sizeof(inp_str->aux_name));
  memset(inp_str->k_grid_file,0,sizeof(inp_str->k_grid_file));
  memset(inp_str->k_grid_file_final,0,sizeof(inp_str->k_grid_file_final));
  memset(inp_str->DFVBfunc,0,sizeof(inp_str->DFVBfunc));
}


inp_info readinp(char *inpname, char* exefile) {

    inp_info inp_str=(inp_info)malloc(sizeof(struct InpInfo));

    FILE *inp_fp;
    FILE *scr_fp;
    FILE *mol_fp;
    FILE *tmp_fp;
    char cline[1024];
    char cline0[1024];
    char str[1024];
    char *comment;
    char *comment1;
    char *comment2;
    char VBDIR0[PATH_MAX];
    char VBDIR[PATH_MAX];
    char basbuf[1024];
    char crdbuf[4][1024];
    char tmpname[1024];
    int natom,ncrd;

  memset(VBDIR,0,sizeof(VBDIR));
  if (realpath(exefile,VBDIR)==NULL) {
    strncpy(VBDIR, exefile, sizeof(VBDIR)-1);
  }
  comment=strstr(VBDIR,"/bin/xeda.exe");
  if (comment==NULL)
    comment=strstr(VBDIR,"/bin/xmvb.exe");
  if (comment!=NULL) {
    memset(comment,0,sizeof(char)*strlen(comment));
  }
  else {
    comment = strstr(VBDIR, "/build-cpp/src/driver/xmvb.exe");
    if (comment == NULL)
      comment = strstr(VBDIR, "/build/src/driver/xmvb.exe");
    if (comment == NULL)
      comment = strstr(VBDIR, "/src/driver/xmvb.exe");
    if (comment != NULL) {
      memset(comment,0,sizeof(char)*strlen(comment));
    }
    else {
      char *last_slash = strrchr(VBDIR, '/');
      if (last_slash != NULL) {
        *last_slash = '\0';
        last_slash = strrchr(VBDIR, '/');
        if (last_slash != NULL) {
          *last_slash = '\0';
        }
      }
    }
  }

//  strcpy(VBDIR,getenv("VBDIR"));
//  comment=getenv("VBDIR");
//  if (comment!=NULL)
//    strcpy(VBDIR,comment);
//  else {
//    printf("Error! Environment variable VBDIR is not set.\n");
//    exit(1);
//  }

  init_inp_param(inp_str);

  //strncpy(inp_str->k_grid_file,VBDIR,strlen(VBDIR));
  strcpy(inp_str->k_grid_file,VBDIR);
  strcpy(inp_str->k_grid_file_final,VBDIR);
  strcat(inp_str->k_grid_file,"/data/kgrids_coarse.opt");
  strcat(inp_str->k_grid_file_final,"/data/kgrids.opt");

  build_runtime_temp_path(inp_str->inpname, sizeof(inp_str->inpname), "scr", ".tmp");
  build_runtime_temp_path(inp_str->mol_name, sizeof(inp_str->mol_name), "mol", ".mol");

  inp_fp=open_required_file(inpname,"r","input file");
  scr_fp=open_required_file(inp_str->inpname,"w+","runtime scratch file");

  const int echo_input_file = should_echo_input_file();
  if (echo_input_file) {
    printf("\n");
    printf("---------------Input File---------------\n");
  }

  memset(inp_str->title,0,sizeof(inp_str->title));

  if (fgets(inp_str->title,1024,inp_fp)==NULL) {
    printf("Error in reading input file.\n");
    exit(1);
  }
  else if (echo_input_file)
    printf("%s",inp_str->title);

// read the whole file, delete the blank lines and comments, punch to scr file
  while(fgets(cline,1024,inp_fp)!=NULL) {

    *(cline+strcspn(cline,"\r\n"))=0;
    if (echo_input_file)
      printf("%s\n",cline);
    comment1=strchr(cline,'#');
    comment2=strchr(cline,';');
    if (comment1==NULL && comment2!=NULL)
      comment=comment2;
    else if (comment2==NULL && comment1!=NULL)
      comment=comment1;
    else if (comment1==NULL && comment2==NULL)
      comment=NULL;
    else {
      if (strlen(comment1)>=strlen(comment2))
        comment=comment1;
      else
        comment=comment2;
    }

    memset(str,0,sizeof(str));
    if (comment==NULL)
      strncpy(str,cline,strlen(cline));
    else
      strncpy(str,cline,strlen(cline)-strlen(comment));

    char *head=get_string_head(str);

    if (head!=NULL) {
      if (should_log_readinp_debug()) {
        fprintf(
            stderr,
            "readinp debug: scr_fp=%p head=%p str='%s'\n",
            (void*)scr_fp,
            (void*)head,
            str);
        fflush(stderr);
      }
      upper_case(head);
      fprintf(scr_fp,"%s\n",head);
    }
  }

  if (echo_input_file) {
    printf("---------------End of Input--------------\n");
    printf("\n");
  }

  fclose(inp_fp);

  if (fvbsec(scr_fp,"$CTR")>0) {
      printf("Error! no $CTR found.\n");
      exit(1);
  }

  // read $CTR section
  getcom(inp_str,scr_fp);

  inp_str->dovb=inp_str->dovbscf+inp_str->dobovb;

  // parse basis set and get aux basis set
  getbasisaux(inp_str);
  strcat(VBDIR,"/basis/");
  memset(basbuf,0,sizeof(basbuf));
  strncpy(basbuf,inp_str->basis_name,strlen(inp_str->basis_name));
  build_prefixed_path_or_die(
      inp_str->basis_name,
      sizeof(inp_str->basis_name),
      VBDIR,
      basbuf);
  memset(basbuf,0,sizeof(basbuf));
  strncpy(basbuf,inp_str->aux_name,strlen(inp_str->aux_name));
  build_prefixed_path_or_die(
      inp_str->aux_name,
      sizeof(inp_str->aux_name),
      VBDIR,
      basbuf);

  if (fvbsec(scr_fp,"$ACTORB") == 0) {
    if (inp_str->orbtyp == GEN_TYP) {
      inp_str->orbtyp = HAO_TYP; 
    }

    if ((inp_str->orbtyp != HAO_TYP) || (inp_str->frgtyp != FRG_ATM)) {
      printf("Error! $ACTORB is only for ORBTYP=HAO and FRGTYP=ATOM.\n");
      printf("Both of the two options are default options and can be omitted if you want to use $ACTORB.\n");
      printf("Please check your input file.\n");
      exit(1);
    }
  }

  if (fvbsec(scr_fp,"$BLWED")==0) {
    parse_blw_params(inp_str,scr_fp);
    if (inp_str->nblw>0)
      inp_str->doblw=1;
      inp_str->doeda=1;
  }

  if (fvbsec(scr_fp,"$BLW")==0) {
    parse_blw_params(inp_str,scr_fp);
    if (inp_str->nblw>0)
      inp_str->doblw=1;
  }

  if (fvbsec(scr_fp,"$EDA")==0) {
    parse_eda_params(inp_str,scr_fp);
    if (inp_str->neda>0)
      inp_str->doeda=1;
  }

  if (inp_str->inttyp!=INT_READ) {
    if (fvbsec(scr_fp,"$GEO")>0) {
        printf("Error! no $GEO found.\n");
        exit(1);
    }

    mol_fp=open_required_file(inp_str->mol_name,"w+","runtime molecule file");
    build_runtime_temp_path(tmpname, sizeof(tmpname), "tmp", ".geo");
    tmp_fp=open_required_file(tmpname,"w","temporary geometry file");

    // get $GEO info and parse .mol file
    natom=0;
    while(fgets(cline,1024,scr_fp)!=NULL) {
        if (strstr(cline,"$END")!=NULL)
            break;
        fprintf(tmp_fp,"%s",cline);
        natom++;
    }

    fclose(tmp_fp);

    tmp_fp=open_required_file(tmpname,"r","temporary geometry file");
    rewind(tmp_fp);

    int nblock=0;
    if (inp_str->doblw>0) 
      nblock=inp_str->nblw;
    else if (inp_str->doeda>0)
      nblock=inp_str->neda;

    for (int i=0;i<3;i++) {
        for (int j=0;j<nblock;j++)
            fprintf(mol_fp," %d",*(inp_str->monomers+i*nblock+j));
        fprintf(mol_fp,"\n");
    }
    fprintf(mol_fp,"%d\n",natom);
    fprintf(mol_fp,"%d %d\n",inp_str->ncharge,inp_str->nmul);

    char** split=malloc_string_array(1024,1024);
    int nsplit=0;
    read_required_line(tmp_fp, cline, 1024, "geometry line");
    memcpy(cline0,cline,sizeof(char)*1024);
//    printf("cline0 =  %s\n",cline0);
//    printf("cline =  %s\n",cline);
    split_string(cline0,split,1024,&nsplit);
//    printf("nsplit = %d\n",nsplit);
    if (nsplit==1) {
//      parse_zmt();
    }
    else {
      if (nsplit==5) {
        fprintf(mol_fp,"%s %s %s %s\n",split[0],split[2],split[3],split[4]);
        for (int i=1;i<natom;i++) {
          read_required_line(tmp_fp, cline, 1024, "geometry line");
          nsplit=0;
          split_string(cline,split,1024,&nsplit);
          fprintf(mol_fp,"%s %s %s %s\n",split[0],split[2],split[3],split[4]);
        }
      }
      else {
        fprintf(mol_fp,"%s\n",cline);
        for (int i=1;i<natom;i++) {
          read_required_line(tmp_fp, cline, 1024, "geometry line");
          fprintf(mol_fp,"%s\n",cline);
        }
      }
    }

    free_string_array(split);

    fclose(tmp_fp);
    remove(tmpname);
    fclose(mol_fp);

  }

  if (inp_str->dowfn>0 && fvbsec(scr_fp,"$AIM")>0) {
      memset(cline,0,sizeof(cline));
      if (fgets(cline,1024,scr_fp)!=NULL) {
          if (strstr(cline,"$END")==NULL)
              strncpy(inp_str->wfn_name,cline,strlen(cline));
      }
  }

  fclose(scr_fp);

  fflush(stdout);

  return inp_str;
}

inp_info readinp_py(char *inpname, char* file_path) {

    inp_info inp_str=(inp_info)malloc(sizeof(struct InpInfo));

    FILE *inp_fp;
    FILE *scr_fp;
    FILE *mol_fp;
    FILE *tmp_fp;
    char cline[1024];
    char cline0[1024];
    char str[1024];
    char *comment;
    char *comment1;
    char *comment2;
//    char VBDIR0[PATH_MAX];
    char VBDIR[PATH_MAX];
    char basbuf[1024];
    char crdbuf[4][1024];
    char tmpname[1024];
    int natom,ncrd;


    strcpy(VBDIR,file_path);

//  memset(VBDIR,0,sizeof(char)*strlen(VBDIR));
//  realpath(exefile,VBDIR);
//  comment=strstr(VBDIR,"/bin/xeda.exe");
//  if (comment==NULL)
//    comment=strstr(VBDIR,"/bin/xmvb.exe");
//  memset(comment,0,sizeof(char)*strlen(comment));

//  strcpy(VBDIR,getenv("VBDIR"));
//  comment=getenv("VBDIR");
//  if (comment!=NULL)
//    strcpy(VBDIR,comment);
//  else {
//    printf("Error! Environment variable VBDIR is not set.\n");
//    exit(1);
//  }

  init_inp_param(inp_str);

  //strncpy(inp_str->k_grid_file,VBDIR,strlen(VBDIR));
  strcpy(inp_str->k_grid_file,VBDIR);
  strcpy(inp_str->k_grid_file_final,VBDIR);
  strcat(inp_str->k_grid_file,"/data/kgrids_coarse.opt");
  strcat(inp_str->k_grid_file_final,"/data/kgrids.opt");

  build_runtime_temp_path(inp_str->inpname, sizeof(inp_str->inpname), "scr", ".tmp");
  build_runtime_temp_path(inp_str->mol_name, sizeof(inp_str->mol_name), "mol", ".mol");

  inp_fp=open_required_file(inpname,"r","input file");
  scr_fp=open_required_file(inp_str->inpname,"w+","runtime scratch file");

  const int echo_input_file = should_echo_input_file();
  memset(inp_str->title,0,sizeof(inp_str->title));

  if (fgets(inp_str->title,1024,inp_fp)==NULL) {
    printf("Error in reading input file.\n");
    exit(1);
  }

// read the whole file, delete the blank lines and comments, punch to scr file
  while(fgets(cline,1024,inp_fp)!=NULL) {

    *(cline+strcspn(cline,"\r\n"))=0;
    comment1=strchr(cline,'#');
    comment2=strchr(cline,';');
    if (comment1==NULL && comment2!=NULL)
      comment=comment2;
    else if (comment2==NULL && comment1!=NULL)
      comment=comment1;
    else if (comment1==NULL && comment2==NULL)
      comment=NULL;
    else {
      if (strlen(comment1)>=strlen(comment2))
        comment=comment1;
      else
        comment=comment2;
    }

    memset(str,0,sizeof(str));
    if (comment==NULL)
      strncpy(str,cline,strlen(cline));
    else
      strncpy(str,cline,strlen(cline)-strlen(comment));

    char *head=get_string_head(str);

    if (head!=NULL) {
      if (should_log_readinp_debug()) {
        fprintf(
            stderr,
            "readinp_py debug: scr_fp=%p head=%p str='%s'\n",
            (void*)scr_fp,
            (void*)head,
            str);
        fflush(stderr);
      }
      upper_case(head);
      fprintf(scr_fp,"%s\n",head);
    }
  }

  if (fvbsec(scr_fp,"$CTR")>0) {
      printf("Error! no $CTR found.\n");
      exit(1);
  }

  // read $CTR section
  getcom(inp_str,scr_fp);

  if (echo_input_file && inp_str->print_level>0) {
    rewind(inp_fp);
    printf("\n");
    printf("---------------Input File---------------\n");
    while(fgets(cline,1024,inp_fp)!=NULL)
      printf("%s",cline);

    printf("---------------End of Input--------------\n");
    printf("\n");
  }

  fclose(inp_fp);

  inp_str->dovb=inp_str->dovbscf+inp_str->dobovb;

  // parse basis set and get aux basis set
  getbasisaux(inp_str);
  strcat(VBDIR,"/basis/");
  memset(basbuf,0,sizeof(basbuf));
  strncpy(basbuf,inp_str->basis_name,strlen(inp_str->basis_name));
  build_prefixed_path_or_die(
      inp_str->basis_name,
      sizeof(inp_str->basis_name),
      VBDIR,
      basbuf);
  memset(basbuf,0,sizeof(basbuf));
  strncpy(basbuf,inp_str->aux_name,strlen(inp_str->aux_name));
  build_prefixed_path_or_die(
      inp_str->aux_name,
      sizeof(inp_str->aux_name),
      VBDIR,
      basbuf);

  if (fvbsec(scr_fp,"$ACTORB") == 0) {
    if (inp_str->orbtyp == GEN_TYP) {
      inp_str->orbtyp = HAO_TYP; 
    }

    if ((inp_str->orbtyp != HAO_TYP) || (inp_str->frgtyp != FRG_ATM)) {
      printf("Error! $ACTORB is only for ORBTYP=HAO and FRGTYP=ATOM.\n");
      printf("Both of the two options are default options and can be omitted if you want to use $ACTORB.\n");
      printf("Please check your input file.\n");
      exit(1);
    }
  }

  if (fvbsec(scr_fp,"$BLWED")==0) {
    parse_blw_params(inp_str,scr_fp);
    if (inp_str->nblw>0)
      inp_str->doblw=1;
      inp_str->doeda=1;
  }

  if (fvbsec(scr_fp,"$BLW")==0) {
    parse_blw_params(inp_str,scr_fp);
    if (inp_str->nblw>0)
      inp_str->doblw=1;
  }

  if (fvbsec(scr_fp,"$EDA")==0) {
    parse_eda_params(inp_str,scr_fp);
    if (inp_str->neda>0)
      inp_str->doeda=1;
  }

  if (inp_str->inttyp!=INT_READ) {
    if (fvbsec(scr_fp,"$GEO")>0) {
        printf("Error! no $GEO found.\n");
        exit(1);
    }

    mol_fp=open_required_file(inp_str->mol_name,"w+","runtime molecule file");
    build_runtime_temp_path(tmpname, sizeof(tmpname), "tmp", ".geo");
    tmp_fp=open_required_file(tmpname,"w","temporary geometry file");

    // get $GEO info and parse .mol file
    natom=0;
    while(fgets(cline,1024,scr_fp)!=NULL) {
        if (strstr(cline,"$END")!=NULL)
            break;
        fprintf(tmp_fp,"%s",cline);
        natom++;
    }

    fclose(tmp_fp);

    tmp_fp=open_required_file(tmpname,"r","temporary geometry file");
    rewind(tmp_fp);

    int nblock=0;
    if (inp_str->doblw>0) 
      nblock=inp_str->nblw;
    else if (inp_str->doeda>0)
      nblock=inp_str->neda;

    for (int i=0;i<3;i++) {
        for (int j=0;j<nblock;j++)
            fprintf(mol_fp," %d",*(inp_str->monomers+i*nblock+j));
        fprintf(mol_fp,"\n");
    }
    fprintf(mol_fp,"%d\n",natom);
    fprintf(mol_fp,"%d %d\n",inp_str->ncharge,inp_str->nmul);

    char** split=malloc_string_array(1024,1024);
    int nsplit=0;
    read_required_line(tmp_fp, cline, 1024, "geometry line");
    memcpy(cline0,cline,sizeof(char)*1024);
//    printf("cline0 =  %s\n",cline0);
//    printf("cline =  %s\n",cline);
    split_string(cline0,split,1024,&nsplit);
//    printf("nsplit = %d\n",nsplit);
    if (nsplit==1) {
//      parse_zmt();
    }
    else {
      if (nsplit==5) {
        fprintf(mol_fp,"%s %s %s %s\n",split[0],split[2],split[3],split[4]);
        for (int i=1;i<natom;i++) {
          read_required_line(tmp_fp, cline, 1024, "geometry line");
          nsplit=0;
          split_string(cline,split,1024,&nsplit);
          fprintf(mol_fp,"%s %s %s %s\n",split[0],split[2],split[3],split[4]);
        }
      }
      else {
        fprintf(mol_fp,"%s\n",cline);
        for (int i=1;i<natom;i++) {
          read_required_line(tmp_fp, cline, 1024, "geometry line");
          fprintf(mol_fp,"%s\n",cline);
        }
      }
    }

    free_string_array(split);

    fclose(tmp_fp);
    remove(tmpname);
    fclose(mol_fp);

  }

  if (inp_str->dowfn>0 && fvbsec(scr_fp,"$AIM")>0) {
      memset(cline,0,sizeof(cline));
      if (fgets(cline,1024,scr_fp)!=NULL) {
          if (strstr(cline,"$END")==NULL)
              strncpy(inp_str->wfn_name,cline,strlen(cline));
      }
  }

  fclose(scr_fp);

  fflush(stdout);

  return inp_str;
}

int get_inttyp_inp_str(inp_info inp_str) { return inp_str->inttyp; }
