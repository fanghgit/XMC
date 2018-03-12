#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include "linear.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct feature_node *x;
long long max_nr_attr = 64;

struct model* model_;
//long long flag_predict_probability=0;

void exit_input_error(long long line_num)
{
	fprintf(stderr,"Wrong input format at line %lld\n", line_num);
	exit(1);
}

static char *line = NULL;
static long long max_line_len;

struct feature_node *x_space;
//struct parameter param;
struct problem test_prob;
double *y_space;  //add for multi-label


static char* readline(FILE *input)
{
	long long len;

	if(fgets(line,(int)max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (long long) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


void exit_with_help()
{
	printf(
	"Usage: predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void read_test(const char *filename)
{
  int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	test_prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char,max_line_len);
  test_prob.n = get_nr_feature(model_);
  test_prob.bias = model_->bias;
  //int nr_class = get_nr_class(model_);

	long long totalCount = 0;

  //printf("begin\n");

	while(readline(fp)!=NULL)
	{
    char *plabel;
		long long c=0;
		plabel = strtok(line,",");
		while (plabel != NULL)
		{
			c++;
			plabel = strtok (NULL, ",");
		}
		totalCount = totalCount+c;

		test_prob.l++;
  }

  //printf("test_prob.l:%d\n", test_prob.l);

  rewind(fp);


	while(readline(fp)!=NULL)
	{
		char *p;
		p = strtok(line," ");

		// features
		while(1)
		{
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
			p = strtok (NULL, " ");
		}
	//	elements++; // for bias term

	}

  //printf("n elements: %d\n", elements);

	rewind(fp);

  //prob.bias=bias;  no bias for now
	test_prob.y = Malloc(double *,test_prob.l);

	y_space = Malloc(double, totalCount);
	test_prob.numLabels = Malloc(int ,test_prob.l);

	test_prob.x = Malloc(struct feature_node *,test_prob.l);
	x_space = Malloc(struct feature_node, elements+test_prob.l);

  max_index = 0;
	j=0;

  //printf("initialization complete\n");

	long long k=0;
	for(i=0;i<test_prob.l;i++)
	{
    //printf("at line: %d\n", i);
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		test_prob.x[i] = &x_space[j];

		test_prob.y[i] = &y_space[k];

		label = strtok(line," ");

		if(label == NULL){ // empty line
			exit_input_error(i+1);
		}
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(idx == NULL){
				break;
			}

			if(val == NULL){
				break;
			}

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
			{	printf("idx error\n");
				printf("idx: %s \n", idx);
				exit_input_error(i+1);
				// fangh debug
			}
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
			{	printf("value error\n");
				printf("val: %s \n", val);
				exit_input_error(i+1);
			}
			++j;
		}

		char *mlabel;
		int cc=0;
		mlabel = strtok(label,",");
		while (mlabel != NULL)
		  {
			cc++;
		    double labelD = strtod(mlabel,NULL);
		    y_space[k] = labelD;
		    k++;
		    mlabel = strtok (NULL, ",");
		  }
		test_prob.numLabels[i] = cc;

		if(inst_max_index > max_index)
			max_index = inst_max_index;

		if(test_prob.bias >= 0)
			x_space[j++].value = test_prob.bias;

		x_space[j++].index = -1;
	}

	if(test_prob.bias >= 0)
	{
		test_prob.n=max_index+1;
		for(i=1;i<test_prob.l;i++)
			(test_prob.x[i]-2)->index = test_prob.n;
		x_space[j-2].index = test_prob.n;
	}
	else
		test_prob.n=max_index;

  if(test_prob.n != get_nr_feature(model_))
  {
    fprintf(stderr,"test nr_feature doesn't match with train. \n");
    exit(1);
  }

	fclose(fp);
	printf("test read complete!\n");

}


// write prediction
void write_pred(FILE *output, int ** pred, int k, int nr_test)
{
  for(int i=0; i<nr_test; i++)
  {
    for(int j=0; j<k; j++)
    {
      fprintf(output, "%d ", pred[i][j]);
    }
    fprintf(output, "\n");
  }
}


// main

int main(int argc, char **argv)
{
	FILE *input, *output;
	long long i;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			// case 'b':
			// 	flag_predict_probability = atoi(argv[i]);
			// 	break;
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}
	if(i>=argc)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}


	// if((model_=load_model(argv[i+1]))==0)
	// {
	// 	fprintf(stderr,"can't open model file %s\n",argv[i+1]);
	// 	exit(1);
	// }
  clock_t start_time = clock();
  struct feature_node **W = NULL;
  model_ = load_model_stat(argv[i+1]);
  W = load_w(argv[i+1]);
  clock_t t = clock();
  printf("load model complete, time spent: %lf sec\n", (double(t-start_time))/CLOCKS_PER_SEC );
	//x = (struct feature_node *) malloc(max_nr_attr*sizeof(struct feature_node));
	//do_predict(input, output);
  start_time = clock();
  read_test(argv[i]);   // input test file
  t = clock();
  printf("load test file complete, time spent: %lf sec\n", (double(t-start_time))/CLOCKS_PER_SEC );

  int k = 5;
  start = clock();
  int ** pred_label = predict(test_prob.x, model_, W, test_prob.l, k);
  t = clock();
  printf("time spent on prediction: %lf sec\n", (double(t-start_time))/CLOCKS_PER_SEC );

  // write prediction
  write_pred(output, pred_label, k, test_prob.l);

  //int ** pred_label = NULL;
  evaluate(pred_label, &test_prob, k);

  free_and_destroy_model(&model_);
	free(line);
	free(x_space);
	fclose(input);
	fclose(output);
	return 0;
}
