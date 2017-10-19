#include<iostream>
#include<cstdlib>
#include<cstdlib>
#include<cuda.h>
#include<highgui.h>
#include<cv.h>

#define Mask_size  3
#define TILE_SIZE  32
#define BLOCK_SIZE 32
__constant__ char M[Mask_size*Mask_size];

using namespace std;
using namespace cv;

//Mantener los valores y no darles salida del dominio

__device__ unsigned char clamp(int value)//__device__ es llamado por el Kernel
{
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return  value;
}


//Convolution Kernel con memoria global

__global__ void convolution2DGlobalMemKernel(unsigned char *In,char *Mask, unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
{

   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
        if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
        &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
        {
          Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * Mask[i*Mask_Width+j];
        }
       }
   }

   Out[row*Rowimg+col] = clamp(Pvalue);
}

//Convolution Kernel con memoria constante

__global__ void convolution2DConstantMemKernel(unsigned char *In,unsigned char *Out,int Mask_Width,int Rowimg,int Colimg)
 {
   unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
   unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

   int Pvalue = 0;

   int N_start_point_row = row - (Mask_Width/2);
   int N_start_point_col = col - (Mask_Width/2);

   for(int i = 0; i < Mask_Width; i++)
   {
       for(int j = 0; j < Mask_Width; j++ )
       {
         if((N_start_point_col + j >=0 && N_start_point_col + j < Rowimg)
         &&(N_start_point_row + i >=0 && N_start_point_row + i < Colimg))
         {
           Pvalue += In[(N_start_point_row + i)*Rowimg+(N_start_point_col + j)] * M[i*Mask_Width+j];
         }
       }
    }

   Out[row*Rowimg+col] = clamp(Pvalue);
}

//Convolution Kernel con memoria compartida

__global__ void convolution2DSharedMemKernel(unsigned char *imageInput,unsigned char *imageOutput,
 int maskWidth, int width, int height)
{
    __shared__ float N_ds[TILE_SIZE + Mask_size - 1][TILE_SIZE + Mask_size - 1];
    int n = maskWidth/2;
    int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+Mask_size-1), destX = dest % (TILE_SIZE+Mask_size-1),
        srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
        src = (srcY * width + srcX);
    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        N_ds[destY][destX] = imageInput[src];
    else
        N_ds[destY][destX] = 0;

    // Segundo procesamiento por lotes
    dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
    destY = dest /(TILE_SIZE + Mask_size - 1), destX = dest % (TILE_SIZE + Mask_size - 1);
    srcY = blockIdx.y * TILE_SIZE + destY - n;
    srcX = blockIdx.x * TILE_SIZE + destX - n;
    src = (srcY * width + srcX);
    if (destY < TILE_SIZE + Mask_size - 1)
    {
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = imageInput[src];
        else
            N_ds[destY][destX] = 0;
    }
    __syncthreads();

    int Pvalue = 0;
    int y, x;
    for (y = 0; y < maskWidth; y++)
        for (x = 0; x < maskWidth; x++)
            Pvalue += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * maskWidth + x];
    y = blockIdx.y * TILE_SIZE + threadIdx.y;
    x = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (y < height && x < width)
        imageOutput[(y * width + x)] = clamp(Pvalue);
    __syncthreads();
}

//Llamado al Kernel

void convolution2DKernelCall(Mat image,unsigned char *In,unsigned char *Out,char *h_Mask,
  int Mask_Width,int Row,int Col, int op)
{
  // Variables
  int Size_of_bytes =  sizeof(unsigned char)*Row*Col*image.channels();
  int Mask_size_bytes =  sizeof(char)*(Mask_size*Mask_size);
  unsigned char *d_In, *d_Out;
  char *d_Mask;
  float Blocksize=BLOCK_SIZE;


  // Asignación de memoria en Device
  cudaMalloc((void**)&d_In,Size_of_bytes);
  cudaMalloc((void**)&d_Out,Size_of_bytes);
  cudaMalloc((void**)&d_Mask,Mask_size_bytes);

  // Memcpy Host to device
  cudaMemcpy(d_In,In,Size_of_bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(d_Mask,h_Mask,Mask_size_bytes,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(M,h_Mask,Mask_size_bytes);// Using constant mem

  dim3 dimGrid(ceil(Row/Blocksize),ceil(Col/Blocksize),1);
  dim3 dimBlock(Blocksize,Blocksize,1);

  //Para seleccionar cual kernel se quiere ejecutar
  switch(op)
  {
    case 1:
    cout<<"2D convolution usando memoria Global"<<endl;
    convolution2DGlobalMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Mask,d_Out,Mask_Width,Row,Col);
    break;
    case 2:
    cout<<"2D convolution usando memoria Constante"<<endl;
    convolution2DConstantMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Out,Mask_Width,Row,Col);
    break;
    case 3:
    cout<<"2D convolution usando memoria Compartida"<<endl;
    convolution2DSharedMemKernel<<<dimGrid,dimBlock>>>(d_In,d_Out,Mask_Width,Row,Col);
    break;
  }

  cudaDeviceSynchronize();

  // Guarda los resultados de salida
  cudaMemcpy (Out,d_Out,Size_of_bytes,cudaMemcpyDeviceToHost);

  // Memoria liberada Device
  cudaFree(d_In);
  cudaFree(d_Out);
  cudaFree(d_Mask);
}

int main()
{
  //Clock variables
  clock_t start, finish;
  double elapsedParallel;
  double elapsedSequential;
  double optimization;
  int Mask_Width =  Mask_size;
  Mat image;
  image = imread("inputs/img1.jpg",0);   // Lee el archivo, 0 significa que ya existe una imagen cargada en esacala de grises
  Size s = image.size();
  int Row = s.width;
  int Col = s.height;
  char h_Mask[] = {-1,0,1,-2,0,2,-1,0,1};

  unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());
  unsigned char *imgOut = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());

  if( !image.data )
  {
    cout<<"Problema cargando la imágen"<<endl;
    return -1;
  }

  img = image.data;

  for (int i = 0; i<11; i++){

  cout<<"Iteración: "<<i<<endl;
  Mat grad_x;
  start = clock();
  Sobel(image,grad_x,CV_8UC1,1,0,3,1,0,BORDER_DEFAULT);
  finish = clock();
  elapsedSequential = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "El proceso secuencial tardó: " << elapsedSequential << " segundos en ejecutarse "<< endl;
  start = clock();
  convolution2DKernelCall(image,img,imgOut,h_Mask,Mask_Width,Row,Col,1);
  finish = clock();
  elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "El proceso paralelo tardó: " << elapsedParallel << " segundos en ejecutarse "<< endl;
  optimization = elapsedSequential/elapsedParallel;
  cout<< "La aceleración  : " << optimization << "X" <<endl;


  start = clock();
  convolution2DKernelCall(image,img,imgOut,h_Mask,Mask_Width,Row,Col,2);
  finish = clock();
  elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "El proceso paralelo tardó: " << elapsedParallel << " segundos en ejecutarse "<< endl;
	optimization = elapsedSequential/elapsedParallel;
  cout<< "La aceleración  : " << optimization << "X" <<endl;

	start = clock();
  convolution2DKernelCall(image,img,imgOut,h_Mask,Mask_Width,Row,Col,3);
  finish = clock();
  elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
  cout<< "El proceso paralelo tardó: " << elapsedParallel << " segundos en ejecutarse "<< endl;
  optimization = elapsedSequential/elapsedParallel;
  cout<< "La aceleración  : " << optimization << "X" <<endl;
  }
  return 0;
}
