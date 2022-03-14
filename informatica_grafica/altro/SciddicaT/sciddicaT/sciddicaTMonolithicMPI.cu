#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../util.hpp"
#include <cuda.h>
#include <mpi.h>

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

// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((i)*(ADJACENT_CELLS)*(columns)) + ((j)*(ADJACENT_CELLS)) + (n) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((i)*(ADJACENT_CELLS)*(columns)) + ((j)*(ADJACENT_CELLS)) + (n) )] )

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

void addLayer2D_Unified(int rows, int columns, double* array)
{
  cudaMallocManaged(&array, sizeof(double) * rows * columns);
}

double* addLayer2D(int rows, int columns)
{
  double *tmp = (double *)malloc(sizeof(double) * rows * columns);
  if (!tmp)
    return NULL;
  return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
__global__ void sciddicaTSimulationInit_Kernel(int r, int c, double* Sz, double* Sh, int i_start, int i_end, int j_start, int j_end)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;    //righe
  int j = blockIdx.y * blockDim.y + threadIdx.y;    //colonne
  if(j < j_start || j >= j_end)
    return;
  if(i < i_start || i >= i_end)
    return;
  
  double z, h;
  h = GET(Sh, c, i, j);

  if (h > 0.0)
  {
    z = GET(Sz, c, i, j);
    SET(Sz, c, i, j, z - h);
  }
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
__global__ void sciddicaTResetFlows_Kernel(int r, int c, double nodata, double* Sf, int i_start, int i_end, int j_start, int j_end)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < i_start || i >= i_end)
    return;
  if(j < j_start || j >= j_end)
    return;
  
  BUF_SET(Sf, r, c, 0, i, j, 0.0);
  BUF_SET(Sf, r, c, 1, i, j, 0.0);
  BUF_SET(Sf, r, c, 2, i, j, 0.0);
  BUF_SET(Sf, r, c, 3, i, j, 0.0);
}

__global__ void sciddicaTFlowsComputation_Kernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon, int i_start, int i_end, int j_start, int j_end, double* halo_Sh, double* halo_Sz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(j < j_start || j >= j_end)
    return;
  if(i < i_start || i >= i_end)
    return;
  
  bool eliminated_cells[5] = {false, false, false, false, false};
  bool again;
  int cells_count;
  double average;
  double m;
  double u[5];
  int n;
  double z, h;

  m = GET(Sh, c, i, j) - p_epsilon;
  u[0] = GET(Sz, c, i, j) + p_epsilon;

  if(i + Xi[1] == -1)
  {
    z = halo_Sz[j + Xj[1]];
    h = halo_Sh[j + Xj[1]];
  }
  else
  {
    z = GET(Sz, c, i + Xi[1], j + Xj[1]);
    h = GET(Sh, c, i + Xi[1], j + Xj[1]);
  }                                      
  u[1] = z + h;

  z = GET(Sz, c, i + Xi[2], j + Xj[2]);
  h = GET(Sh, c, i + Xi[2], j + Xj[2]);
  u[2] = z + h;

  z = GET(Sz, c, i + Xi[3], j + Xj[3]);
  h = GET(Sh, c, i + Xi[3], j + Xj[3]);
  u[3] = z + h;

  if(i + Xi[4] == r)
  {
    z = halo_Sz[j + Xj[4]];
    h = halo_Sh[j + Xj[4]];
  }
  else
  {
    z = GET(Sz, c, i + Xi[4], j + Xj[4]);
    h = GET(Sh, c, i + Xi[4], j + Xj[4]);
  }                                        
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

__global__ void sciddicaTWidthUpdate_Kernel(int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, int i_start, int i_end, int j_start, int j_end, double* halo_Sf)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if(i < i_start || i >= i_end)
    return;
  if(j < j_start || j >= j_end)
    return;
 
  double h_next;
  h_next = GET(Sh, c, i, j);

  if(i + Xi[1] == -1)
    h_next += halo_Sf[(j+Xj[1])*4 + 3] - BUF_GET(Sf, r, c, 0, i, j);
  else
    h_next += BUF_GET(Sf, r, c, 3, i+Xi[1], j+Xj[1]) - BUF_GET(Sf, r, c, 0, i, j);

  h_next += BUF_GET(Sf, r, c, 2, i+Xi[2], j+Xj[2]) - BUF_GET(Sf, r, c, 1, i, j);
  h_next += BUF_GET(Sf, r, c, 1, i+Xi[3], j+Xj[3]) - BUF_GET(Sf, r, c, 2, i, j);
  
  if(i + Xi[4] == r)
    h_next += halo_Sf[(j+Xj[4])*4 + 0] - BUF_GET(Sf, r, c, 3, i, j);
  else
    h_next += BUF_GET(Sf, r, c, 0, i+Xi[4], j+Xj[4]) - BUF_GET(Sf, r, c, 3, i, j);

  SET(Sh, c, i, j, h_next );
}

// ----------------------------------------------------------------------------
void updateHalo(double* halo_Sh, double* halo_Sz, double* halo_Sf, double* Sh, double* Sz, double* Sf, int r, int c, int pid, bool only_sf)
{
  // Varibili native di MPI di stato e richiesta
  MPI_Request request = MPI_REQUEST_NULL;
  MPI_Status status;

  if(!only_sf)
  {
    if(pid == 0)
    {
      /*
      -Input parameters
      buf: initial address of send buffer
      count: number of elements in send buffer (integer)
      datatype: datatype of each send buffer element (handle)
      dest: rank of destination (integer)
      tag: message tag (integer)
      comm: communicator (handle)
      -Output parameters
      request: communication request (handle)
      */
      // int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
      MPI_Isend(&Sh[(r-1)*c], c, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request);
      /*
      -Input Parameters
      buf: initial address of receive buffer (choice)
      count: number of elements in receive buffer (integer)
      datatype: datatype of each receive buffer element (handle)
      source: rank of source (integer)
      tag: message tag (integer)
      comm: communicator (handle)
      -Output Parameters
      request: communication request (handle)
      */
      // int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
      MPI_Irecv(&halo_Sh[0], c, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &request);
    }
    else if(pid == 1)
    {
      MPI_Irecv(&halo_Sh[0], c, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
      MPI_Isend(&Sh[0], c, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
    }
  }
  else
  {
    if(pid == 0)
    {
      MPI_Isend(&BUF_GET(Sf, r, c, 0, r-1, 0), c*ADJACENT_CELLS, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request);
      MPI_Irecv(&halo_Sf[0], c*ADJACENT_CELLS, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &request);
    }
    else
    {
      MPI_Irecv(&halo_Sf[0], c*ADJACENT_CELLS, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
      MPI_Isend(&BUF_GET(Sf, r, c, 0, 0, 0), c*ADJACENT_CELLS, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
    }
  }

  /*  
  -Input Parameters
  request: request (handle)
  -Output Parameters
  status: status object (Status)
  */
  // int MPI_Wait(MPI_Request *request, MPI_Status *status)
  MPI_Wait(&request, &status);
}

// ----------------------------------------------------------------------------
void compute_process(int r, int c, int steps, double p_r, double p_epsilon, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, double nodata, int pid)
{ 
  int i_start = 1;
  int i_end = r - 1;
  int j_start = 1;
  int j_end = c - 1;

  if(pid == 0)
    i_end = r;
  else if(pid == 1)
    i_start = -1;

  double* halo_Sz;
  double* halo_Sh;
  double* halo_Sf;

  cudaMallocManaged(&halo_Sz, sizeof(double) * c);
  cudaMallocManaged(&halo_Sh, sizeof(double) * c);
  cudaMallocManaged(&halo_Sf, sizeof(double) * c * ADJACENT_CELLS);
  
  dim3 dimGrid((r-1)/16 + 1, (c-1)/16 + 1, 1);
  dim3 dimBlock(16,16,1);

  // Init kernel
  sciddicaTSimulationInit_Kernel<<<dimGrid, dimBlock>>>(r, c, Sz, Sh, i_start, i_end, j_start, j_end);
  cudaDeviceSynchronize();

  MPI_Request request = MPI_REQUEST_NULL;
  MPI_Status status;

  if(pid == 0)
  {
    MPI_Isend(&Sz[(r-1)*c], c, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &request);
    MPI_Irecv(&halo_Sz[0], c, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &request);
  }
  else if(pid == 1)
  { 
    MPI_Irecv(&halo_Sz[0], c, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
    MPI_Isend(&Sz[0], c, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request);
  }

  MPI_Wait(&request, &status);

  util::Timer cl_timer;
  
  // Simulation loop
  for(int s = 0; s < steps; ++s)
  { 
    updateHalo(halo_Sh, halo_Sz, halo_Sf, Sh, Sz, Sf, r, c, pid, false);    // update sh
    
    // ResetFlow kernel
    sciddicaTResetFlows_Kernel<<<dimGrid, dimBlock>>>(r, c, nodata, Sf, i_start, i_end, j_start, j_end);
    cudaDeviceSynchronize();
    
    // FlowComputation kernel
    sciddicaTFlowsComputation_Kernel<<<dimGrid, dimBlock>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon, i_start, i_end, j_start, j_end, halo_Sh, halo_Sz);
    cudaDeviceSynchronize();

    updateHalo(halo_Sh, halo_Sz, halo_Sf, Sh, Sz, Sf, r, c, pid, true);    // update sf

    // WidthUpdate kernel
    sciddicaTWidthUpdate_Kernel<<<dimGrid, dimBlock>>>(r, c, nodata, Xi, Xj, Sz, Sh, Sf, i_start, i_end, j_start, j_end, halo_Sf);
    cudaDeviceSynchronize();
  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  printf("Releasing memory...\n");
  
  cudaFree(Xi);
  cudaFree(Xj);
  cudaFree(halo_Sh);
  cudaFree(halo_Sz);
  cudaFree(halo_Sf);
}


// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  double *Sz;
  double *Sh;
  double *Sf;
  double *Sz_server;
  double *Sh_server;
  int *Xi;
  int *Xj;

  int r = rows;                     // r: grid rows
  int c = cols;                     // c: grid columns
  double p_r = P_R;                 // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;     // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]); //steps: simulation steps
  int pid = -1;
  int np = -1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  // Select the GPU according to the pid
  cudaSetDevice(pid);

  int r_local = r / np;

  cudaMallocManaged(&Sz, sizeof(double) * r_local * c);
  cudaMallocManaged(&Sh, sizeof(double) * r_local * c);
  cudaMallocManaged(&Sf, sizeof(double) * ADJACENT_CELLS * r_local * c);

  if(pid == 0)
  {
    Sz_server = (double *)malloc(sizeof(double) * r * c);
    Sh_server = (double *)malloc(sizeof(double) * r * c);

    loadGrid2D(Sz_server, r, c, argv[DEM_PATH_ID]);    // Load Sz from file
    loadGrid2D(Sh_server, r, c, argv[SOURCE_PATH_ID]); // Load Sh from file
  }

  /*
  -Input Parameters
  sendbuf: address of send buffer (choice, significant only at root)
  sendcount: number of elements sent to each process (integer, significant only at root)
  sendtype: data type of send buffer elements (significant only at root) (handle)
  recvcount: number of elements in receive buffer (integer)
  recvtype: data type of receive buffer elements (handle)
  root: rank of sending process (integer)
  comm: communicator (handle)
  -Output Parameters
  recvbuf: address of receive buffer (choice)
  */
  // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
  MPI_Scatter(Sz_server, r_local*c, MPI_DOUBLE, Sz, r_local*c, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(Sh_server, r_local*c, MPI_DOUBLE, Sh, r_local*c, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

  compute_process(r_local, c, steps, p_r, p_epsilon, Xi, Xj, Sz, Sh, Sf, nodata, pid);

  /*
  -Input Parameters
  sendbuf: starting address of send buffer (choice)
  sendcount: number of elements in send buffer (integer)
  sendtype: data type of send buffer elements (handle)
  recvcount: number of elements for any single receive (integer, significant only at root)
  recvtype: data type of recv buffer elements (significant only at root) (handle)
  root: rank of receiving process (integer)
  comm: communicator (handle)
  -Output Parameters
  recvbuf: address of receive buffer (choice, significant only at root)
  */
  // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
  MPI_Gather(Sz, r_local*c, MPI_DOUBLE, Sz_server, r_local*c, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(Sh, r_local*c, MPI_DOUBLE, Sh_server, r_local*c, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(pid == 0)
  {
    saveGrid2Dr(Sh_server, r, c, argv[OUTPUT_PATH_ID]);   // Save Sh to file
    delete[] Sz_server;
    delete[] Sh_server;
  }

  /*
  -Input Parameters
  comm: communicator (handle)
  */
  // int MPI_Barrier( MPI_Comm comm )
  MPI_Barrier(MPI_COMM_WORLD);

  cudaFree(Sz);
  cudaFree(Sh);
  cudaFree(Sf);
  
  MPI_Finalize();

  return 0;
}