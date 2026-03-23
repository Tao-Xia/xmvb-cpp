#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Move the file position indicator to the first line where str first appears
int fvbsec(FILE *fp,char *str) {

    char line[1024];
    rewind(fp);
    while (fgets(line,1024,fp)!=NULL) {
        if (strstr(line,str)!=NULL) {
          return 0;
        }
    }

    return 1;
}

