#include <stdio.h>
#include <stdlib.h>

char** malloc_string_array(size_t nstring, size_t nlen) {
  char** string=(char**)malloc(sizeof(char*)*nstring);
  string[0]=(char*)calloc(nstring*nlen,sizeof(char));
  for (int i=1;i<nstring;i++)
    string[i]=string[i-1]+nlen;

  return string;
}

void free_string_array(char** string) {
  free(string[0]);
  free(string);
  string=NULL;
  return;
}
