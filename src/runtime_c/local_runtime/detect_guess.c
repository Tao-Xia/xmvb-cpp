#include <stdio.h>
#include "vb/vb.h"
#include "inpout/input.h"

int detect_guess(char *inpname) {

    char cline[2014]; 


    FILE *fp = fopen(inpname, "r"); 

    if (fvbsec(fp,"$GUS") > 0) {
        // printf("Error! No $GUS found.\n"); 
        // exit(1); 
        return 0; 
    }

    int guesstype = GUS_MO; 

    char** split=malloc_string_array(1024,1024);
    int nsplit;
    while(fgets(cline, 1024, fp) != NULL) {
        if (strstr(cline, "$END") != NULL) {
            break; 
        }

        if (strstr(cline, ".") != NULL) {
            guesstype = GUS_READ; 
            break; 
        }

        nsplit = 0; 
        split_string(cline, split, 1024,&nsplit); 
        if (nsplit != 2) {
            guesstype = GUS_READ; 
            break; 
        }
    }

    free_string_array(split);

    return guesstype; 
} 
