
#include "GPUTiledMatMultPlugin.h"

void GPUTiledMatMultPlugin::input(std::string infile) {
   readParameterFile(infile);
}

void GPUTiledMatMultPlugin::run() {}

void GPUTiledMatMultPlugin::output(std::string outfile) {
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

int M, N, P;
 M = atoi(myParameters["M"].c_str());
 N = atoi(myParameters["N"].c_str());
 P = atoi(myParameters["P"].c_str());
 numARows = M;
 numAColumns = N;
 numBRows = N;
 numBColumns = P;
 numCRows = M;
 numCColumns = P;

  hostA = (float*) malloc (M*N*sizeof(float));
  hostB = (float*) malloc (N*P*sizeof(float));
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numARows * numBColumns * sizeof(float));

  numCRows    = numARows;
  numCColumns = numBColumns;

 std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["matrix1"]).c_str(), std::ios::in);
 int i;
 for (i = 0; i < M*N; ++i) {
        float k;
        myinput >> k;
        hostA[i] = k;
 }
 std::ifstream myinput2((std::string(PluginManager::prefix())+myParameters["matrix2"]).c_str(), std::ios::in);
 for (i = 0; i < N*P; ++i) {
        float k;
        myinput2 >> k;
        hostB[i] = k;
 }



  //@@ Allocate GPU memory here
  cudaMalloc(&deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc(&deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc(&deviceC, sizeof(float) * numCRows * numCColumns);
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns,
             cudaMemcpyHostToDevice);


  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid((numCColumns - 1) / TILE_WIDTH + 1,
               (numCRows - 1) / TILE_WIDTH + 1, 1);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid, dimBlock>>>(
      deviceA, deviceB, deviceC, numARows, numAColumns, numBRows,
      numBColumns, numCRows, numCColumns);

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns,
             cudaMemcpyDeviceToHost);

  std::ofstream outsfile(outfile.c_str(), std::ios::out);

        for (i = 0; i < M*P; ++i){
                outsfile << hostC[i];//std::setprecision(0) << a[i*N+j];
                outsfile << "\n";
        }


  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);


  free(hostA);
  free(hostB);
  free(hostC);

}

PluginProxy<GPUTiledMatMultPlugin> GPUTiledMatMultPluginProxy = PluginProxy<GPUTiledMatMultPlugin>("GPUTiledMatMult", PluginManager::getInstance());
