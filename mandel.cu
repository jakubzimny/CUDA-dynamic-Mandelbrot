#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <png.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID 65535

struct Complex{
    double re;
    double im;
};

__device__ Complex multiply_complex(Complex c1, Complex c2){
    Complex temp;
    temp.re = c1.re * c2.re - c1.im * c2.im;
    temp.im = c1.im * c2.re + c1.re * c2.im;
    return temp;
}

__device__ Complex abs_complex(Complex c){
    Complex temp;
    temp.re = abs(c.re);
    temp.im = abs(c.im);
    return temp;
}

__device__ double magnitude_squared(Complex c){
    return c.re * c.re + c.im * c.im;
}

__device__ int get_color_value(int iter){
    // Some arbitrary color values for now, can be improved with linear coloring algorithm
    int val = iter + 2; // value is increased to make the image a bit brighter
    // R G B values based on iter, scaled by arbitrary coefficents to get some aesthetic colors
    int color = 0 | (val * 2) | (val * 4 << 8) | ((val * 10) << 16);
    return color;
}

__global__ void render_mandelbrot_k(int h, int w, int max_iter, int *results){
    long my_index = blockIdx.x * blockDim.x + threadIdx.x;
    double a, b;
    Complex c1, c2;
    int x = my_index % w;
    int y = my_index / w;
    a = ((double)x - ((double)w / 2.0)) * 4.0 / (double)w;
    b = ((double)y - ((double)h / 2.0)) * 4.0 / (double)w;
    c1 = {0.0, 0.0};
    c2 = {a, b};
    int iter = 0;
    while (magnitude_squared(c1) < 4 && iter++ < max_iter){
        c1 = multiply_complex(c1, c1);
        c1.re += c2.re;
        c1.im += c2.im;
    }
    if(iter < max_iter) results[my_index] = get_color_value(iter);
    else results[my_index] = 0;
}

__host__ void error_exit(const char *error_message){
    printf("%s\n", error_message);
    exit(-1);
}

__host__ void save_as_png(char *filename, int width, int height, int* pixel_values) {
    FILE *fp = fopen(filename, "wb");
    png_bytep png_row = NULL;
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    setjmp(png_jmpbuf(png_ptr));
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height,
        8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_text title_text;
    title_text.compression = PNG_TEXT_COMPRESSION_NONE;
    title_text.key = "Title";
    title_text.text = "Mandelbrot set visualization";
    png_set_text(png_ptr, info_ptr, &title_text, 1);
    png_write_info(png_ptr, info_ptr);

    png_row = (png_bytep) malloc(sizeof(png_byte) * width * 3);

    for (int row = 0; row < height; row++){
        for(int col = 0; col < width; col++){
            png_row[col * 3] = (png_byte)(pixel_values[row * width + col] & 0xFF); // R
            png_row[col * 3 + 1] = (png_byte)((pixel_values[row * width + col] >> 8) & 0xFF); // G
            png_row[col * 3 + 2] = (png_byte)((pixel_values[row * width + col] >> 16) & 0xFF); // B
        }
        png_write_row(png_ptr, png_row);
    }

    png_write_end(png_ptr, info_ptr);
    fclose(fp);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(png_row);
}

__host__ void parse_paramaters(int argc, char* argv[], int* height, int* width, int* max_iter, char** filename){
    if (argc > 1){
        for(int i = 1; i < argc; i+=2){
            if(argv[i+1] != NULL && argv[i+1][0] != '-'){
                if (strcmp(argv[i], "-o") == 0){
                    *filename = argv[i+1];
                }
                else if(strcmp(argv[i], "-w") == 0){
                    int w = atoi(argv[i+1]);
                    if (w > 0) *width = w;
                    else error_exit("Invalid width value.");
                }
                else if(strcmp(argv[i], "-h") == 0){
                    int h = atoi(argv[i+1]);
                    if (h > 0) *height = h;
                    else error_exit("Invalid height value.");
                }
                else if(strcmp(argv[i], "-iter") == 0){
                    int iter = atoi(argv[i+1]);
                    if (iter > 0) *max_iter = iter;
                    else error_exit("Invalid number of iterations.");
                }
                else
                    error_exit("Invalid input paramaters");
            }
        else error_exit("No value supplied for argument");
        }
    }
}

int main (int argc, char* argv[]){
    struct timeval start, end;

    //default values
    int width = 1024;
    int height = 1024;
    int max_iter = 1000;
    char* filename = "Mandelbrot.png";
    parse_paramaters(argc, argv, &height, &width, &max_iter, &filename);
    long long pixel_count = (long long)width * (long long)height;
    // program is designed for each thread to compute one pixel, so pixel count cannot exceed maximum thread count
    if (pixel_count > (MAX_THREADS_PER_BLOCK * MAX_BLOCKS_PER_GRID)){
        error_exit("Requested resolution is not supported. Try lowering height and/or width values.");
    }

    int threads_in_block = MAX_THREADS_PER_BLOCK;
    int blocks_in_grid = (int)ceil((double)(height * width) / (double)threads_in_block);

    gettimeofday(&start, NULL);

    long size = height * width * sizeof(double);
    int* host_result = (int*)malloc(size);
    if (!host_result){
        error_exit("Error while allocating memory on the host, exiting.");
    }
    int* device_result = NULL;
    if (cudaSuccess != cudaMalloc((void **)&device_result, size)){
        error_exit("Error while allocating memory on the device, exiting.");
    }

    render_mandelbrot_k<<<blocks_in_grid, threads_in_block>>>(height,
                                                            width,
                                                            max_iter,
                                                            device_result);

    if (cudaSuccess != cudaMemcpy(host_result, device_result, size, cudaMemcpyDeviceToHost)){
       error_exit("Error while copying results from device to host, exiting.");
    }

    gettimeofday(&end, NULL);
    long long elapsed = 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    printf("GPU calculations took %lld microseconds.\n", elapsed);

    save_as_png(filename, width, height, host_result);

    free(host_result);
    if (cudaSuccess != cudaFree(device_result)){
        error_exit("Error while deallocating memory on the device, exiting.");
    }
}
