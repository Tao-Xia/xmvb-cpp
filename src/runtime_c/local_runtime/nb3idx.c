#include <stdio.h>
#include "vb/vb.h"

int build_nb3idx(vb_info vb_str) {

  if (vb_str->nb3num!=NULL) {
    free(vb_str->nb3num);
    free(vb_str->nb3idx);
  }
  long nb = (long)vb_str->nb;
  long n2e = (long)vb_str->n2e;
//  printf("n2e = %ld\n", n2e); 
//  printf("nb = %ld\n", nb); 
//  fflush(stdout); 
  vb_str->nb3idx=(long *)malloc(sizeof(long)*n2e*nb);
//  printf("sizeof(nb3idx) = "); 
//  printf("%ld\n", sizeof(long)*n2e*nb); 
//  printf("Mem need is %ldGB\n", sizeof(long)*n2e*nb/(1024*1024*1024)); 
//  fflush(stdout); 
  vb_str->nb3num=(long *)malloc(sizeof(long)*nb);
//  printf("sizeof(nb3num) = "); 
//  printf("%ld\n", sizeof(long)*nb); 
//  fflush(stdout); 
  // printf("here you find it%ld,%ld",nb,n2e);
  // exit(0);
  memset(vb_str->nb3num,0,sizeof(long)*nb);
  memset(vb_str->nb3idx,0,sizeof(long)*n2e*nb);
  int idx,i,j,k,l;
  for (idx=0;idx<vb_str->n2e;idx++) {
    i=*(vb_str->g2eidx+idx*4  );
    j=*(vb_str->g2eidx+idx*4+1);
    k=*(vb_str->g2eidx+idx*4+2);
    l=*(vb_str->g2eidx+idx*4+3);
    *(vb_str->nb3idx+i*n2e+(*(vb_str->nb3num+i)))=idx;
    (*(vb_str->nb3num+i))++;
    if (j!=i) {
      *(vb_str->nb3idx+j*n2e+(*(vb_str->nb3num+j)))=idx;
      (*(vb_str->nb3num+j))++;
    }
    if (k!=i && k!=j) {
      *(vb_str->nb3idx+k*n2e+(*(vb_str->nb3num+k)))=idx;
      (*(vb_str->nb3num+k))++;
    }
    if (l!=i && l!=j && l!=k) {
      *(vb_str->nb3idx+l*n2e+(*(vb_str->nb3num+l)))=idx;
      (*(vb_str->nb3num+l))++;
    }
  }

  return 0;
}

void build_nb3idx_(vb_info vb_str) {
  build_nb3idx(vb_str);
}
