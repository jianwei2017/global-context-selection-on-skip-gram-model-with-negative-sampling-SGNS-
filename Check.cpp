#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
using namespace std;

#define MAX_STRING 1000
#define MAX_SENTENCE_LENGTH 1000
#define max_vocab_size 200000

const int vocab_hash_size = 30000000;
int vocab_max_size = 1000;
const int maxn_size = 100000;
int top = 10;

struct node{
    int id;
    double w;

    bool operator <(const node &b)const{
        return w > b.w;
    }
};

struct vocab_word{
    char *word;
    int type;
    vector<node>V;
    vector<node>V2;
};

char train_file[MAX_STRING],check_file[MAX_STRING],output_file[MAX_STRING],read_file[MAX_STRING];
int vocab_size = 0;
int *vocab_hash;
struct vocab_word *vocab;
double weight;
double *vocab_vector;


int GetWordHash(char *word)
{
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

int SearchVocab(char *word)
{
    unsigned int hash = GetWordHash(word);
    while (1)
    {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

int AddWordToVocab(char *word)
{
    unsigned int hash, length = strlen(word) + 2;
    //if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab_size++;
    if (vocab_size + 2 >= vocab_max_size)
    {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

void Generate()
{
    FILE *fin, *fo;
    char eof;
    char *s1, *s2, *tmp;
    s1 = (char *)calloc(MAX_SENTENCE_LENGTH,sizeof(char));
    s2 = (char *)calloc(MAX_SENTENCE_LENGTH,sizeof(char));
    tmp = (char *)calloc(MAX_SENTENCE_LENGTH,sizeof(char));
    printf("Starting reading file %s\n",train_file);

    fin = fopen(train_file,"rb");
    if(fin == NULL)
    {
        printf("ERROR: the input data file not found!\n");
        exit(1);
    }

    for(int i=0;i<vocab_hash_size;i++) vocab_hash[i] = -1;

    int total = 3000;
    while(total--)
    {
        fscanf(fin, "%s%s%lf", s1, s2, &weight);
        int lens1 = strlen(s1);
        int lens2 = strlen(s2);
        int cnt=0;
        for(int i=0;i<lens1;i++){
            if(s1[i]=='-') break;
            tmp[cnt++]=s1[i];
        }
        tmp[cnt++]='\0';
        int id1 = SearchVocab(tmp);
        if(id1 == -1){
            id1 = AddWordToVocab(tmp);
            vocab[id1].V.clear();
            vocab[id1].type = 0;
        }
        cnt=0;
        for(int i=0;i<lens2;i++){
            if(s2[i]=='-') break;
            tmp[cnt++]=s2[i];
        }
        tmp[cnt++]='\0';
        int id2 = SearchVocab(tmp);
        if(id2 == -1){
            id2 = AddWordToVocab(tmp);
            vocab[id2].V.clear();
            vocab[id2].type = 0;
        }
        vocab[id1].V.push_back(node{id2,weight});
        vocab[id2].V.push_back(node{id1,weight});
    }
    fclose(fin);

    printf("Reading Finished\n");
    if(output_file[0]!=0){

        printf("Starting save file %s\n",output_file);

        fo = fopen(output_file,"wb");
        fprintf(fo,"%d\n",vocab_size);

        for(int a=0;a<vocab_size;a++){
            sort(vocab[a].V.begin(),vocab[a].V.end());
            fprintf(fo,"%s %ld ", vocab[a].word, vocab[a].V.size());
            for(int b=0;b<vocab[a].V.size();b++){
                fprintf(fo,"%s ",vocab[vocab[a].V[b].id].word);
            }
            fprintf(fo,"\n");
        }
        fclose(fo);
    }
}

void Read_File(){
    FILE *fin;
    int total;
    int vnum;
    char *s1, *s2;
    s1 = (char *)calloc(MAX_SENTENCE_LENGTH,sizeof(char));
    s2 = (char *)calloc(MAX_SENTENCE_LENGTH,sizeof(char));
    fin = fopen(read_file,"rb");

    if(fin == NULL){
        printf("ERROR: the sequence data file not found!\n");
        exit(1);
    }

    for(int i=0;i<vocab_hash_size;i++) vocab_hash[i] = -1;

    fscanf(fin, "%d", &total);
    while(total--){
        fscanf(fin, "%s %d", s1, &vnum);
        int id1 = SearchVocab(s1);
        if(id1 == -1){
            id1 = AddWordToVocab(s1);
            vocab[id1].V.clear();
            vocab[id1].type = 0;
        }

        for(int i = 0; i < vnum; i++) {
            fscanf(fin, "%s", s2);
            int id2 = SearchVocab(s2);
            if(id2 == -1){
                id2 = AddWordToVocab(s2);
                vocab[id2].V.clear();
                vocab[id2].type = 0;
            }
            vocab[id1].V.push_back(node{id2,(vnum-i)*1.0});
        }
    }
    printf("Reading Finished!\n");
    fclose(fin);
}

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

void Check(){
    FILE *fch;
    char *s1, *s2;
    int id1, id2;
    double ans;
    int total,vec_size,test_size;
    s1 = (char *)calloc(MAX_SENTENCE_LENGTH,sizeof(char));
    s2 = (char *)calloc(MAX_SENTENCE_LENGTH,sizeof(char));
    if(read_file[0] == 0) Generate();
    else Read_File();

    ans = 0;
    fch = fopen(check_file,"rb");
    if(fch == NULL){
        printf("ERROR: the check file not found!\n");
        exit(1);
    }
    fscanf(fch,"%d %d",&test_size,&vec_size);


    printf("\tVocab size : %d Vector size : %d\n",vocab_size,vec_size);
    vocab_vector = (double *)calloc(vec_size * (vocab_size + 1) + 500, sizeof(double));
    for(int i=0;i<vec_size * (vocab_size + 1) + 500;i++) vocab_vector[i] = -1;

    double tmp;
    for(int i=0;i<test_size;i++){
        fscanf(fch,"%s",s1);
        id1 = SearchVocab(s1);
        if(id1 == -1) for(int j=0;j<vec_size;j++) fscanf(fch,"%lf",&tmp);
        else for(int j=0;j<vec_size;j++) fscanf(fch,"%lf",&vocab_vector[id1*vec_size+j]);
    }

    fclose(fch);
    printf("Reading Check file finished!\n");
    int cnt2=0;

    ans = 0;
    for(int i=0;i<vocab_size;i++){
        if(vocab_vector[i*vec_size]==-1)continue;
        else {
            for(int j=0;j<vocab[i].V.size();j++){
                int id = vocab[i].V[j].id;
                if(vocab_vector[id*vec_size] == -1) break;
                vocab[i].V2.push_back(node{id,cosine(i,id,vec_size)});
            }
            sort(vocab[i].V2.begin(),vocab[i].V2.end());

            int s1,s2;
            s1 = 0, s2 = 0;
            int cnt=0,Count=0;
            double res=0;
            while(s1 < vocab[i].V.size() && s2 < vocab[i].V2.size() && cnt <= top){
                int id1 = vocab[i].V[s1].id;
                int id2 = vocab[i].V2[s2].id;
                if(vocab_vector[id1*vec_size] == -1) s1++;
                else if(vocab_vector[id2*vec_size] == -1) s2++;
                else {
                    Count++;
                    if(id1 == id2) {
                        cnt++;
                        res += cnt * 1.0 / Count;
                    }
                    s1++;
                    s2++;
                }
            }
            cnt2++;
            if(cnt) res /= cnt;
            ans += res;
        }
    }
    ans /= cnt2;
    printf("\tThe Mean Average Precision is : %lf\n",ans);
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
        printf("\t\tUse data from <file> to generate the squence.\n");
        printf("\t-check <file>\n");
        printf("\t\tUse data from <file> to check the squence.\n");
        printf("\t-output <file>\n");
        printf("\t\tprint generated squence to <file>.\n");
        printf("\t-seqread <file>\n");
        printf("\t\tread the generated sequence.\n");
        printf("\t-top <int>\n");
        printf("\t\tset the top threshold, default is 10.\n");
        return 0;
    }

    output_file[0] = 0;
    read_file[0] = 0;

    if ((i = ArgPos((char *)"-input", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-check", argc, argv)) > 0) strcpy(check_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-seqread", argc, argv)) > 0) strcpy(read_file,argv[i + 1]);
    if ((i = ArgPos((char *)"-top", argc, argv)) > 0) top = atoi(argv[i + 1]);

    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    vocab = (struct vocab_word *)calloc(vocab_max_size,sizeof(struct vocab_word));

    Check();

    return 0;
}
