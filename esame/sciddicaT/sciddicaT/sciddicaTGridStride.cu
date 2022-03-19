#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.hpp"
#include <cuda.h>

using namespace std;

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define P_R 0.5
#define P_EPSILON 0.001
#define ADJACENT_CELLS 4
#define STRLEN 256

#define BLOCK_SIZE 16

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
void readHeaderInfo(char* path, int &nrows, int &ncols, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata)
{
  FILE* f;
  
  if ( (f = fopen(path,"r") ) == 0){
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  //Reading the header
  char str[STRLEN];
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); ncols = atoi(str);      //ncols
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nrows = atoi(str);      //nrows
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str);  //xllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str);  //yllcorner
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);   //cellsize
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);     //NODATA_value 
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
__global__ void sciddicaTSimulationInit_Kernel(int r, int c, double* Sz, double* Sh, int i_start, int i_end, int j_start, int j_end)
{
  // riga e colonna, se il numero di thread è inferiore alla dimensione della matrice, il thread calcola più celle
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < r; i += blockDim.x * gridDim.x)
  {
    for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < c; j += blockDim.y * gridDim.y) 
    {
      double z, h;
      if(i < i_start || i >= i_end)
        continue;
      if(j < j_start || j >= j_end)
        continue;
      
      h = GET(Sh, c, i, j);

      if (h > 0.0)
      {
        z = GET(Sz, c, i, j);
        SET(Sz, c, i, j, z - h);
      }
    } 
  }
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__global__ void sciddicaTResetFlows_Kernel(int r, int c, double nodata, double* Sf, int i_start, int i_end, int j_start, int j_end)
{
  // riga e colonna, se il numero di thread è inferiore alla dimensione della matrice, il thread calcola più celle
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < r; i += blockDim.x * gridDim.x)
  {
    for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < c; j += blockDim.y * gridDim.y)
    {
      if(i < i_start || i >= i_end)
        continue;
      if(j < j_start || j >= j_end)
        continue;
      
      BUF_SET(Sf, r, c, 0, i, j, 0.0);
      BUF_SET(Sf, r, c, 1, i, j, 0.0);
      BUF_SET(Sf, r, c, 2, i, j, 0.0);
      BUF_SET(Sf, r, c, 3, i, j, 0.0);
    }
  }
}

__global__ void sciddicaTFlowsComputation_Kernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon, int i_start, int i_end, int j_start, int j_end)
{
  // riga e colonna, se il numero di thread è inferiore alla dimensione della matrice, il thread calcola più celle
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < r; i += blockDim.x * gridDim.x)
  {
    for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < c; j += blockDim.y * gridDim.y) 
    {
      bool eliminated_cells[5] = {false, false, false, false, false};
      bool again;
      int cells_count;
      double average;
      double m;
      double u[5];
      int n;
      double z, h;
      if(i < i_start || i >= i_end)
        continue;
      if(j < j_start || j >= j_end)
        continue;
      
      m = GET(Sh, c, i, j) - p_epsilon;
      u[0] = GET(Sz, c, i, j) + p_epsilon;
      z = GET(Sz, c, i + Xi[1], j + Xj[1]);
      h = GET(Sh, c, i + Xi[1], j + Xj[1]);
      u[1] = z + h;                                         
      z = GET(Sz, c, i + Xi[2], j + Xj[2]);
      h = GET(Sh, c, i + Xi[2], j + Xj[2]);
      u[2] = z + h;                                         
      z = GET(Sz, c, i + Xi[3], j + Xj[3]);
      h = GET(Sh, c, i + Xi[3], j + Xj[3]);
      u[3] = z + h;                                         
      z = GET(Sz, c, i + Xi[4], j + Xj[4]);
      h = GET(Sh, c, i + Xi[4], j + Xj[4]);
      u[4] = z + h;

      do
      {
        again = false;
        average = m;
        cells_count = 0;

        for (n = 0; n < 5; n++)
          if (!eliminated_cells[n])
          {
            average += u[n];
            cells_count++;
          }

        if (cells_count != 0)
          average /= cells_count;

        for (n = 0; n < 5; n++)
          if ((average <= u[n]) && (!eliminated_cells[n]))
          {
            eliminated_cells[n] = true;
            again = true;
          }
      } while (again);

      if (!eliminated_cells[1]) BUF_SET(Sf, r, c, 0, i, j, (average - u[1]) * p_r);
      if (!eliminated_cells[2]) BUF_SET(Sf, r, c, 1, i, j, (average - u[2]) * p_r);
      if (!eliminated_cells[3]) BUF_SET(Sf, r, c, 2, i, j, (average - u[3]) * p_r);
      if (!eliminated_cells[4]) BUF_SET(Sf, r, c, 3, i, j, (average - u[4]) * p_r);
    }
  }
}

__global__ void sciddicaTWidthUpdate_Kernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, int i_start, int i_end, int j_start, int j_end)
{
  // riga e colonna, se il numero di thread è inferiore alla dimensione della matrice, il thread calcola più celle
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < r; i += blockDim.x * gridDim.x)
  {
    for(int j = blockIdx.y * blockDim.y + threadIdx.y; j < c; j += blockDim.y * gridDim.y) 
    {
      double h_next;
      if(i < i_start || i >= i_end)
        continue;
      if(j < j_start || j >= j_end)
        continue;

      h_next = GET(Sh, c, i, j);

      h_next += BUF_GET(Sf, r, c, 3, i+Xi[1], j+Xj[1]) - BUF_GET(Sf, r, c, 0, i, j);
      h_next += BUF_GET(Sf, r, c, 2, i+Xi[2], j+Xj[2]) - BUF_GET(Sf, r, c, 1, i, j);
      h_next += BUF_GET(Sf, r, c, 1, i+Xi[3], j+Xj[3]) - BUF_GET(Sf, r, c, 2, i, j);
      h_next += BUF_GET(Sf, r, c, 0, i+Xi[4], j+Xj[4]) - BUF_GET(Sf, r, c, 3, i, j);
      
      SET(Sh, c, i, j, h_next );
    }
  }
}


// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows;                  // r: grid rows
  int c = cols;                  // c: grid columns
  int i_start = 1, i_end = r-1;  // [i_start,i_end[: kernels application range along the rows
  int j_start = 1, j_end = c-1;  // [i_start,i_end[: kernels application range along the rows
  double *Sz;                    // Sz: substate (grid) containing the cells' altitude a.s.l.
  double *Sh;                    // Sh: substate (grid) containing the cells' flow thickness
  double *Sf;                    // Sf: 4 substates containing the flows towards the 4 neighs
  int *Xi;                       // Xi: von Neuman neighborhood row coordinates (see below)
  int *Xj;                       // Xj: von Neuman neighborhood col coordinates (see below)
  double p_r = P_R;                 // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;     // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); //steps: simulation steps

  cudaMallocManaged(&Xi, sizeof(int) * 5);
  cudaMallocManaged(&Xj, sizeof(int) * 5);

  Xi[0] = 0;
  Xi[1] = -1;
  Xi[2] = 0;
  Xi[3] = 0;
  Xi[4] = 1;

  Xj[0] = 0;
  Xj[1] = 0;
  Xj[2] = -1;
  Xj[3] = 1;
  Xj[4] = 0;

  cudaMallocManaged(&Sz, sizeof(double) * r * c);
  cudaMallocManaged(&Sh, sizeof(double) * r * c);
  cudaMallocManaged(&Sf, sizeof(double) * ADJACENT_CELLS * r * c);

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);      // Load Sz from file
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);   // Load Sh from file

  dim3 dimGrid(ceil(r/(float)BLOCK_SIZE), ceil(c/(float)BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  // Init kernel
  sciddicaTSimulationInit_Kernel<<<dimGrid, dimBlock>>>(r, c, Sz, Sh, i_start, i_end, j_start, j_end);
  cudaDeviceSynchronize();

  util::Timer cl_timer;

  // Simulation loop
  for (int s = 0; s < steps; ++s)
  {
    // ResetFlow kernel
    sciddicaTResetFlows_Kernel<<<dimGrid, dimBlock>>>(r, c, nodata, Sf, i_start, i_end, j_start, j_end);
    cudaDeviceSynchronize();

    // FlowComputation kernel
    sciddicaTFlowsComputation_Kernel<<<dimGrid, dimBlock>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon, i_start, i_end, j_start, j_end);
    cudaDeviceSynchronize();

    // WidthUpdate kernel
    sciddicaTWidthUpdate_Kernel<<<dimGrid, dimBlock>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, i_start, i_end, j_start, j_end);
    cudaDeviceSynchronize();
  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);  // Save Sh to file

  printf("Releasing memory...\n");
  
  cudaFree(Sz);
  cudaFree(Sh);
  cudaFree(Sf);
  
  cudaFree(Xi);
  cudaFree(Xj);

  return 0;
}