#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_WINDOW_SIZE 11
#define PATH_LEN 1000
const int vocab_hash_size = 30000000;

struct node
{
    int wordid;
    int stringid;
    node *next;
};

struct vocab_word
{
    long long cn;
    long long cwu;
    double w;
    char *word;
    node *next;
};

char train_file[MAX_STRING],output_file[MAX_STRING],Dir_name[MAX_STRING];
char Union_file[MAX_STRING];
char save_vocab_file[MAX_STRING],read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
struct vocab_word *pair;

int window = 5,min_count = 5,num_threads = 12,num_reduce =1,min_reduce=1;
int debug_mode = 2;
int *vocab_hash;
long long vocab_max_size=1000,vocab_size=0,layer1_size=100;
long long train_words = 0, word_count_actual = 0, iter = 5,file_size = 0;
double alpha=0.025,starting_alpha,sample=1e-3;
double *syn0, *syn1, *syn1neg, *expTable;
clock_t start;
int negative = 5;
const int table_size = 1e8;
int *table;

int methodnum;
long long pairnum;
int *Delete;
int *pair_hash;
long long pair_size=0;
char words[MAX_WINDOW_SIZE][MAX_STRING];
double totalw=0;
long long pair_max_size=10000;
double *weights;
double min_weight;
double threshold;
FILE *funion;


void InitUnigramTable()
{
    int a, i;
    double train_words_pow = 0;
    double d1, power = 0.75;
    table = (int *)malloc(table_size * sizeof(int));
    if(methodnum == 1) for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
    if(methodnum >= 2) for (a = 0; a < vocab_size; a++) train_words_pow += vocab[a].w;
    i = 0;
    if(methodnum == 1) d1 = pow(vocab[i].cn, power) / train_words_pow;
    if(methodnum >= 2) d1 = vocab[i].w / train_words_pow;
    for (a = 0; a < table_size; a++)
    {
        table[a] = i;
        if (a / (double)table_size > d1)
        {
            i++;
            if(methodnum == 1) d1 += pow(vocab[i].cn, power) / train_words_pow;
            if(methodnum >= 2) d1 += vocab[i].w / train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1;
    }
}

void ReadWord(char *word, FILE *fin, char *eof)
{
    int a = 0, ch;
    while (1)
    {
        ch = fgetc_unlocked(fin);
        if (ch == EOF)
        {
            *eof = 1;
            break;
        }
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n'))
        {
            if (a > 0)
            {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n')
            {
                strcpy(word, (char *)"</s>");
                return;
            }
            else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

int GetWordHash(char *word)
{
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

int SearchVocab(char *word, int type)
{
    unsigned int hash = GetWordHash(word);
    while (1)
    {
        if(type == 1)
        {
            if (vocab_hash[hash] == -1) return -1;
            if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
	    hash = (hash + 1) % vocab_hash_size;
        }
        else if(type == 2)
        {
            if (pair_hash[hash] == -1) return -1;
            if (!strcmp(word, pair[pair_hash[hash]].word)) return pair_hash[hash];
	    hash = (hash + 1) % (vocab_hash_size << 3);
        }
    }
    return -1;
}

int ReadWordIndex(FILE *fin, char *eof)
{
    char word[MAX_STRING], eof_l = 0;
    ReadWord(word, fin, &eof_l);
    if (eof_l)
    {
        *eof = 1;
        return -1;
    }
    return SearchVocab(word, 1);
}

int AddWordToVocab(char *word,int type)
{
    unsigned int hash, length = strlen(word) + 2;
    //if (length > MAX_STRING) length = MAX_STRING;
    if(type==1)
    {
        vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
        strcpy(vocab[vocab_size].word, word);
        vocab[vocab_size].cn = 0;
        vocab[vocab_size].next = NULL;
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
    else if(type == 2)
    {
        pair[pair_size].word = (char *)calloc(length, sizeof(char));
        strcpy(pair[pair_size].word, word);
        pair[pair_size].cn = 0;
        pair[pair_size].next = NULL;
        pair_size++;
        if (pair_size + 2 >= pair_max_size)
        {
            pair_max_size += 1000;
            pair = (struct vocab_word *)realloc(pair, pair_max_size * sizeof(struct vocab_word));
        }
        hash = GetWordHash(word);
        while (pair_hash[hash] != -1) hash = (hash + 1) % (vocab_hash_size << 3);
        pair_hash[hash] = pair_size - 1;
        return pair_size - 1;
    }
}

int VocabCompare(const void *a, const void *b)
{
    long long l = ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
    if (l > 0) return 1;
    if (l < 0) return -1;
    return 0;
}
int WeightsCompare(const void *a, const void *b)
{
    return (*(double *)b) - (*(double *)a);
}

void SortVocab()
{
    Delete = (int *)calloc(vocab_size,sizeof(int));
    for(int i=0;i<vocab_size;i++) Delete[i] = 0;
    if(methodnum == 1)
    {
        int a, size;
        unsigned int hash;
        qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
        for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
        size = vocab_size;
        train_words = 0;
        for (a = 0; a < size; a++)
        {
            if ((vocab[a].cn < min_count) && (a != 0))
            {
                vocab_size--;
                free(vocab[a].word);
            }
            else
            {
                hash=GetWordHash(vocab[a].word);
                while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
                vocab_hash[hash] = a;
                train_words += vocab[a].cn;
            }
        }
        vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    }
    else if(methodnum >= 2)
    {
        int a, size;
        unsigned int hash;
        qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
        for (a = 0; a < vocab_hash_size; a++) vocab_hash[a]  = -1;
        size = vocab_size;
        train_words = 0;
        for (a = 0; a < size; a++)
        {
            if ((vocab[a].cn < min_count) && (a != 0))
            {
                Delete[a]=1;
                vocab_size--;
                node *tem,*cur;
                tem = vocab[a].next;
                while(tem!=NULL){
                    cur = tem;
                    tem = tem->next;
                    free(cur);
                }
                free(vocab[a].word);
            }
            else
            {
                hash = GetWordHash(vocab[a].word);;
                while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
                vocab_hash[hash] = a;
                train_words += vocab[a].cn;
            }
        }
        vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    }
}

void ReduceVocab()
{
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce)
        {
            vocab[b].cn = vocab[a].cn;
            vocab[b].word = vocab[a].word;
            b++;
        }
        else {
		node *NEXT, *tem;
		NEXT = vocab[a].next;
		while(NEXT != NULL)
		{
			tem = NEXT;
			NEXT = NEXT->next;
			free(tem);
		}	
		free(vocab[a].word);
	}
    vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 0; a < vocab_size; a++)
    {
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}


void InitNet()
{
    long long a, b;
    unsigned long long next_random = 1;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(double));
    if (syn0 == NULL)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(double));
    if (syn1neg == NULL)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
            syn1neg[a * layer1_size + b] = 0;

    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
        {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (double)65536) - 0.5) / layer1_size;
        }
}

void CalWeight()
{
    int i;
    double pw, pu, puw, pu_w, p_w;
    node *tmp, *tmp2;
    long long A, B, C, D, N;
    weights = (double *)malloc(vocab_size * sizeof(double));

    int position = floor(threshold * vocab_size);

    for(i = 0; i < vocab_size; i++)
    {
        //printf("wordid : %d\n",i);
        pw = vocab[i].cn * 1.0 / train_words;
        tmp = vocab[i].next;
        double CHI,MI;
        while(tmp != NULL)
        {
            // the word is deleted
            if(Delete[tmp->wordid] == 1)
            {
                tmp2 = tmp;
                tmp = tmp->next;
                free(tmp2);
                continue;
            }
            //printf("to wordid: %d word: %s\n",tmp->wordid,vocab[tmp->wordid].word);
            pu = vocab[tmp->wordid].cn * 1.0 / train_words;
            // IG
            if(methodnum == 2)
            {
                p_w = 1 - pw;
                puw = (pair[tmp->stringid].cn * 1.0 / pairnum) / pw;
                pu_w = ((vocab[tmp->wordid].cwu - pair[tmp->stringid].cn ) * 1.0 / pairnum) / p_w;
                vocab[i].w += - pu*log(pu) + pw * puw * log(puw) + p_w * pu_w * log(pu_w);
            }
            // MI
            if(methodnum == 3)
            {
                A = pair[tmp->stringid].cn;
                B = vocab[i].cwu - pair[tmp->stringid].cn;
                C = vocab[tmp->wordid].cwu - pair[tmp->stringid].cn;
                N = pairnum;
                MI = log(A * N * 1.0 / ((A + C) * (A + B)));
                vocab[i].w += pu * MI;
            }
            // CHI
            if(methodnum == 4)
            {
                A = pair[tmp->stringid].cn;
                B = vocab[i].cwu - A;
                C = vocab[tmp->wordid].cwu - A;
                D = pairnum - vocab[tmp->wordid].cwu - vocab[i].cwu + A;
                N = pairnum;
                CHI = (N * (A * D - C * B) * (A * D - C * B) * 1.0) / ((A + C) * (B + D) * (A + B) * (C + D));
                vocab[i].w += pu * CHI;
            }
            tmp2 = tmp;
            free(tmp2);
            tmp = tmp->next;
        }
        weights[i] = vocab[i].w;
        totalw += vocab[i].w;
    }
    qsort(&weights[1], vocab_size-1, sizeof(double), WeightsCompare);
    min_weight = weights[position];
    printf("Calculate Finished!\n");
}

void Create_Table(char *filename){
    char word[MAX_STRING], eof = 0;
    char tmp[(MAX_SENTENCE_LENGTH<<2) + 1];
    char tmp2[(MAX_SENTENCE_LENGTH<<2) + 1];
    long long a, wid, j,uid,w_uid,u_wid;
    FILE *fin;
    fin = fopen64(filename, "rb");
    if (fin == NULL)
    {
        printf("ERROR: %s not found!\n",filename);
        exit(1);
    }
    long long cnt=0;

    while (1)
    {
        ReadWord(word, fin, &eof);
        if(Union_file[0] != 0)
	{
	    fprintf(funion,"%s ",word);
        };
        if (eof) break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0))
        {
            printf("words:%lldKvaocab:%lldpairs:%lld%c", train_words / 1000, vocab_size, cnt, 13);
            fflush(stdout);
        }
        wid = SearchVocab(word, 1);
        if (wid == -1)
        {
            a = AddWordToVocab(word, 1);
            vocab[a].cn = 1;
            wid = a;
        }
        else vocab[wid].cn++;

        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
        if (methodnum >= 2)
        {
            if(train_words > window)
            {
                for(j = 1; j <= window; j++)
                {
                    //to is the id of u in words arrary
                    int to = (train_words % window - j + window ) % window;

                    //build w_u pair
                    strcpy(tmp, word);
                    strcpy(tmp + strlen(tmp), "_");
                    strcpy(tmp + strlen(tmp), words[to]);

                    //build u_w pair
                    strcpy(tmp2, words[to]);
                    strcpy(tmp2 + strlen(tmp2), "_");
                    strcpy(tmp2 + strlen(tmp2), word);

                    wid = SearchVocab(word, 1);
                    vocab[wid].cwu++;

                    uid = SearchVocab(words[to], 1);
                    if(uid == -1)
                    {
                        uid = AddWordToVocab(words[to], 1);
                        vocab[uid].cwu++;
                    }
                    else vocab[uid].cwu++;

                    w_uid = SearchVocab(tmp, 2);
                    if (w_uid == -1)
                    {
                        w_uid = AddWordToVocab(tmp, 2);
                        pair[w_uid].cn = 1;
                        cnt++;
                        node *NEXT = (struct node*)malloc(sizeof(struct node));
                        NEXT->next = vocab[wid].next;
                        vocab[wid].next = NEXT;
                        NEXT->wordid = uid;
                        NEXT->stringid = w_uid;
                    }
                    else pair[w_uid].cn++;

                    u_wid = SearchVocab(tmp2, 2);
                    if (u_wid == -1)
                    {
                        u_wid = AddWordToVocab(tmp2, 2);
                        pair[u_wid].cn = 1;
                        cnt++;
                        node *NEXT = (struct node*)malloc(sizeof(struct node));
                        NEXT->next = vocab[uid].next;
                        vocab[uid].next = NEXT;
                        NEXT->wordid = wid;
                        NEXT->stringid = u_wid;
                    }
                    else pair[u_wid].cn++;

                    pairnum += 2;
                }
            }
            strcpy(words[train_words % window], word);
        }
    }
    file_size += ftell(fin);
    fclose(fin);

}

void dir_scan(const char *path, const char *file)
{
    struct stat s;
    DIR *dir;
    struct dirent *dt;
    char dirname[PATH_LEN];
    char THEFILE[PATH_LEN];

    memset(dirname, 0, PATH_LEN*sizeof(char));
    strcpy(dirname, path);


    if(stat(file, &s) < 0)
    {
        printf("lstat error\n ");
        exit(1);
    }

    if(S_ISDIR(s.st_mode))
    {
        strcpy(dirname+strlen(dirname), file);
        strcpy(dirname+strlen(dirname), "/");
        if((dir = opendir(file)) == NULL)
        {
            printf( "opendir %s/%s error\n ",dirname,file);
            exit(1);
        }

        if(chdir(file) < 0)
        {
            printf( "chdir error\n ");
            exit(1);
        }

        while((dt = readdir(dir)) != NULL)
        {
            if(dt-> d_name[0] == '.')
            {
                continue;
            }
            dir_scan(dirname, dt-> d_name);
        }

        if(chdir( "..") < 0)
        {
            printf( "chdir error\n ");
            exit(1);
        }
    }
    else
    {
        strcpy(THEFILE,file);
        Create_Table(THEFILE);
    }

}

void LearnVocabFromTrainFile()
{
    char curdir[PATH_LEN];
    long long a;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 0; a < vocab_hash_size << 3; a++) pair_hash[a] = -1;
    vocab_size = 0;
    pair_size = 0;
    pairnum = 0;
    AddWordToVocab((char *)"</s>", 1);

    if(Union_file[0] == 0 && train_file[0] == 0){
        printf("Union file doesn't exist!");
        exit(0);
    }
    if(Union_file[0]!=0)funion = fopen64(Union_file,"wb");
    
    if(Dir_name[0] == 0){
        Create_Table(train_file);
    }else {
        getcwd(curdir,sizeof(curdir));
        dir_scan("",Dir_name);
        if(chdir(curdir)<0)
        {
            printf( "chdir error\n ");
            exit(1);
        }

    }
    if(Union_file[0] != 0)fclose(funion);
    SortVocab();
    printf("SortVocab!!\n");

    if(methodnum >= 2)
    {
        CalWeight();
        printf("CalWeight!!\n");
    }
    if (debug_mode > 0)
    {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
}

void SaveVocab()
{
    long long i;
    FILE *fo = fopen64(save_vocab_file, "wb");
    for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab()
{
    long long a, i = 0;
    char c, eof = 0;
    char word[MAX_STRING];
    FILE *fin = fopen64(read_vocab_file, "rb");
    if (fin == NULL)
    {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    vocab_size = 0;
    while (1)
    {
        ReadWord(word, fin,&eof);
        if (eof) break;
        a = AddWordToVocab(word, 1);
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);
        i++;
    }
    SortVocab();
    if (debug_mode > 0)
    {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fin = fopen64(train_file, "rb");
    if (fin == NULL)
    {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    file_size += ftell(fin);
    fclose(fin);
}

void *TrainModel(void *id)
{
    long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
    long long l1, l2, c, target, label, local_iter = iter;
    unsigned long long next_random = (long long)id;
    double f, g;
    clock_t now;
    char eof = 0;
    double *neu1 = (double *)calloc(layer1_size, sizeof(double));
    double *neu1e = (double *)calloc(layer1_size, sizeof(double));
    FILE *fi;
    if(Union_file[0] == 0)
    {
	fi = fopen64(train_file,"rb");
    }
    else fi = fopen64(Union_file, "rb");
    fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

    while(1)
    {
        if (word_count - last_word_count > 10000)
        {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;
            now = clock();
            printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                   word_count_actual / (double)(iter * train_words + 1) * 100,
                   word_count_actual / ((double)(now - start + 1) / (double)CLOCKS_PER_SEC * 1000));
            fflush(stdout);
        }
        //auto-modify learning rate
        alpha = starting_alpha * (1 - word_count_actual / (double)(iter * train_words + 1));
        if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;

        if(sentence_length==0)
        {
            while (1)
            {
                word = ReadWordIndex(fi,&eof);
                if (eof) break;
                if (word == -1) continue;
                if (methodnum >= 2) if (vocab[word].w < min_weight) continue;
                word_count++;
                if (word == 0) break;
                double ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                next_random = next_random * (unsigned long long)25214903917 + 11;
                if (ran < (next_random & 0xFFFF) / (double)65536) continue;
                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }

        if (eof || (word_count > train_words / num_threads))
        {
            word_count_actual += word_count - last_word_count;
            local_iter--;
            if (local_iter == 0) break;
            word_count = 0;
            last_word_count = 0;
            sentence_length = 0;
            fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
            continue;
        }

        word = sen[sentence_position];
        if (word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] = 0;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        next_random = next_random * (unsigned long long)25214903917 + 11;
        b = next_random % window;

        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) 
        {
            c = sentence_position - window + a;
            if (c < 0) continue;
            if (c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;
            l1 = last_word * layer1_size;
            for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

            for (d = 0; d < negative + 1; d++)
            {
                if (d == 0)
                {
                    target = word;
                    label = 1;
                }
                else
                {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (vocab_size - 1) + 1;
                    if (target == word) continue;
                    label = 0;
                }
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
            }
            for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }
        sentence_position++;
        if (sentence_position >= sentence_length)
        {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
    free(neu1);
    free(neu1e);
    pthread_exit(NULL);
}

void Train()
{
    long a,b,c,d;
    FILE *fo;

    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    if(Dir_name[0] == 0) printf("Starting training using file %s\n", train_file);
    else printf("Starting training using files in %s\n", Dir_name);
    starting_alpha = alpha;

    printf("LearnVocabFromTrainFile\n");
    if (read_vocab_file[0] != 0) ReadVocab();
    else LearnVocabFromTrainFile();
    printf("Learning Finished\n");

    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;

    InitNet();
    InitUnigramTable();
    start = clock();
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModel, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);


    fo = fopen64(output_file, "wb");

    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++)
    {
        fprintf(fo, "%s ", vocab[a].word);
        for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
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

int methodchoice(char *methodname)
{
    if(strcmp(methodname,"DF") == 0) return 1;
    if(strcmp(methodname,"IG") == 0) return 2;
    if(strcmp(methodname,"MI") == 0) return 3;
    if(strcmp(methodname,"CHI") == 0) return 4;
    return -1;
}


int main(int argc,char **argv)
{
    int i;
    struct stat s;

    if (argc == 1)
    {
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-dir <dir>\n");
        printf("\t\tUse all data from <dir> to train the model\n");
        printf("\t-union <file>\n");
        printf("\t\tUnion all data from <dir> to <file>\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words,range from 5 to 10; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-method\n");
        printf("\t\tSet the feature selection method; default is document frequency thresholding(DF)\n");
        printf("\t\tThe optional parameters are: DF -- document frequency, IG -- Imformation Gain, \n\t\tMI -- Mutual infomation, CHI -- CHI-test\n");
        printf("\t-threshold\n");
        printf("\t\tSet the filter threshold; default is 0.7; the range is 0 - 1\n");
        printf("\nExamples:\n");
        printf("./sgns.exe -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -method DF\n");
        printf("./sgns.exe -dir Data -output vec.txt -size 200 -method IG -threshold 0.9\n\n");
        return 0;
    }

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    Union_file[0] = 0;
    Dir_name[0] = 0;
    methodnum = 1;
    threshold = 0.7;

    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-method",argc,argv)) > 0) methodnum = methodchoice(argv[i + 1]);
    if ((i = ArgPos((char *)"-threshold",argc,argv)) > 0) threshold = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-union",argc,argv)) > 0) strcpy(Union_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-dir",argc,argv)) > 0 ) {
            strcpy(Dir_name,argv[i+1]);
            if(stat(Dir_name, &s) < 0){
                printf("lstat error!\n");
                return 0;
            }
            if(!S_ISDIR(s.st_mode))
            {
                printf("%s is not a dir name\n",Dir_name);
                return 0;
            }
    }

    if(methodnum < 0 )
    {
        printf("This method doesn't exist!\n");
        return 0;
    }
    if(threshold < 0 || threshold > 1)
    {
        printf("The threshold is beyond range!\n");
        return 0;
    }

    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    pair = (struct vocab_word *)calloc(pair_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    pair_hash = (int *)calloc(vocab_hash_size << 3, sizeof(int));
    expTable = (double *)malloc((EXP_TABLE_SIZE + 1) * sizeof(double));

    for (i = 0; i < EXP_TABLE_SIZE; i++)
    {
        expTable[i] = exp((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
        expTable[i] = expTable[i] / (expTable[i] + 1);
    }
    Train();
    printf("Training Finished\n");
    return 0;
}
