#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
using namespace std;

#define MAX_STRING 1000
#define MAX_SENTENCE_LENGTH 1000
#define vocab_max_size 10000

struct similar{
    double val;
    int id;

    bool operator < (const similar &b) const{
        return val > b.val;
    }
};

struct vocab_word{
    char *word;
    vector<similar>v;
};

char train_file[MAX_STRING],output_file2[MAX_STRING];
struct vocab_word *vocab;
double *vocab_vector;
int vocab_size = 0;
int vector_size = 0;

double cosine(int a,int b,int vector_size){
    double tem1=0,tem2=0;
    double denominator = 0;
    double numerator = 0;
    for(int k=0;k<vector_size;k++){
        tem1+=vocab_vector[k+vector_size*a]*vocab_vector[k+vector_size*a];
        tem2+=vocab_vector[k+vector_size*b]*vocab_vector[k+vector_size*b];
        numerator+=vocab_vector[k+vector_size*a]*vocab_vector[k+vector_size*b];
    }
    denominator = sqrt(tem1)*sqrt(tem2);
    return numerator/denominator;
}

void Do_Cal(int vocab_size,int vector_size){
    for(int i=0;i<vocab_size;i++){
        for(int j=i+1;j<vocab_size;j++){
            double val = cosine(i,j,vector_size);
            vocab[i].v.push_back(similar{val,j});
            vocab[j].v.push_back(similar{val,i});
        }
        sort(vocab[i].v.begin(),vocab[i].v.end());
    }
}

void Cal_Dis(){
    FILE *fin;
    printf("Starting calculating using file %s\n",train_file);

    fin = fopen(train_file,"rb");
    if(fin == NULL)
    {
        printf("ERROR: the input data file not found!\n");
        exit(1);
    }

    char c;
    fscanf(fin,"%d%d%c",&vocab_size,&vector_size,&c);
    printf("\tVocab size: %d\n\tVector size: %d\n",vocab_size,vector_size);

    vocab = (struct vocab_word *)calloc(vocab_size + 100,sizeof(struct vocab_word));
    vocab_vector=(double *)calloc(vector_size * vocab_size + 100 ,sizeof(double));

    int word_count = 0;
    while(word_count<vocab_size)
    {
        vocab[word_count].word = (char *)calloc(MAX_SENTENCE_LENGTH,sizeof(char));
        fscanf(fin,"%s",vocab[word_count].word);
        for(int i=0;i<vector_size;i++){
            fscanf(fin,"%lf",&vocab_vector[word_count*vector_size+i]);
        }
        word_count++;
    }
    fclose(fin);
    printf("Reading Finish\n");
    Do_Cal(vocab_size,vector_size);
    printf("Calculate Finish\n");

    if(output_file2[0] != 0){
        FILE *fo = fopen(output_file2,"wb");
        fprintf(fo,"%d %d\n",vocab_size,vector_size);
        for(int a=0;a<vocab_size;a++){
            fprintf(fo,"%s ",vocab[a].word);
            for(int b=0;b<100;b++){
                fprintf(fo, "%s ",vocab[vocab[a].v[b].id].word);
            }
            fprintf(fo,"\n");
        }
        fclose(fo);
    }
    printf("Output Finish\n");
    while(1){
        char word[MAX_SENTENCE_LENGTH];
        printf("\tInput a word to show similar words: (input 'exit()' to exit)");
        scanf("%s",word);
        if(strcmp(word,"exit()")==0) break;

        int IS_FOUND=0;
        for(int i=0;i<vocab_size;i++){
            if(strcmp(word,vocab[i].word)==0){
                IS_FOUND = 1;
                for(int j=0;j<10;j++){
                    printf("\t%d\t\t%s %lf\n",j+1,vocab[vocab[i].v[j].id].word,vocab[i].v[j].val);
                }
                printf("\t......\n");
                break;
            }
        }
        if(IS_FOUND==0){
            printf("the word doesn't exist!\n");
        }
        fflush(stdout);
    }
    free(vocab_vector);
    free(vocab);
}


int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a]))
        {
            if (a == argc - 1)
            {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

int main(int argc,char **argv){
    int i;
    if(argc==1){
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-input <file>\n");
        printf("\t\tUse text data from <file> to calculate the distance\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the similar word\n");
        return 0;
    }
    output_file2[0] = 0;

    if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file2, argv[i + 1]);

    Cal_Dis();
    return 0;
}
