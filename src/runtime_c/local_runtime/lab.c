#include <stdio.h>

int lab(int i, int j) {
  if (i>j)
    return i*(i+1)/2+j;
  else
    return j*(j+1)/2+i;
}

long lab_long(long i, long j) {
  if (i > j) 
      return i*(i+1)/2 + j; 
  else 
      return j*(j+1)/2 + i; 
}

int lab_(int *i, int *j) {
  return lab((*i)-1,(*j)-1)+1;
}

long lab_long_(long *i, long *j) {
  return lab_long((*i)-1,(*j)-1)+1;
}
