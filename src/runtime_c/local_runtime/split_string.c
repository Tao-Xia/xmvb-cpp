#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Split the line "string" into several strings by " " and store in "split" and store the number of keywords in nsplit 
void split_string(char *string, char** split, int nlen, int *nsplit) {
    int istart,iend,clen,idx,slen;

    slen=strlen(string); // Get the length of string 
    istart=-2;
    iend=-1;
    idx=0;
    while (idx<slen) {
      if (*(string+idx)!=' ' && istart <= iend){
        istart=idx;
        if ((idx == (slen-1)) && (*(string+idx) != '\r') && (*(string+idx) != '\n')) {
          memset(split[*nsplit], 0, sizeof(char)*nlen); 
          memcpy(split[*nsplit], string+istart, sizeof(char)); 
          if (strlen(split[*nsplit]) > 0) {
            int ilen=strlen(split[*nsplit]);
            if (split[*nsplit][ilen-1]=='\r' || split[*nsplit][ilen-1]=='\n') {
              split[*nsplit][ilen-1]=='\0';
            }
            if (strlen(split[*nsplit])>0)
              (*nsplit)++; 
          }
        }
      }
      else if (iend < istart)
        if (*(string+idx)==' ' || idx==slen-1) {
          iend=idx;
          if (*(string+idx)==' ')
            iend--;
          clen=iend-istart+1;
          memset(split[*nsplit],0,sizeof(char)*nlen);
          memcpy(split[*nsplit],string+istart,clen*sizeof(char));
          if (strlen(split[*nsplit])>0)
            (*nsplit)++;
        }
      idx++;
    }

    return;
}

