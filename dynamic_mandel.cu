#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <png.h>

#define INITIAL_SQUARES 64
#define BLOCK_DIVISION 4
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define MAX_RECURSION_DEPTH 4

#define MIN_SIZE 32 * 32

#define MIN(a,b) (((a)<(b))?(a):(b))

struct Complex{
    double re;
    double im;
};


__device__ __host__ Complex multiply_complex(Complex c1, Complex c2){
    Complex temp;
    temp.re = c1.re * c2.re - c1.im * c2.im;
    temp.im = c1.im * c2.re + c1.re * c2.im;
    return temp;
}


__device__ __host__ Complex abs_complex(Complex c){
    Complex temp;
    temp.re = abs(c.re);
    temp.im = abs(c.im);
    return temp;
}


__device__ __host__ double magnitude_squared(Complex c){
    return c.re * c.re + c.im * c.im;
}


__device__ __host__ int div_round_up(int a, int b){
    return (int)ceil((double)a / (double)b);
}


__device__ __host__ int get_color_value(int iter){
/* Returns color value based on iter */
    int val = iter + 2; // value is increased to make the image a bit brighter
    // R G B values based on iter, scaled by arbitrary coefficents to get some aesthetic colors
    int color = 0 | (val * 2) | (val * 4 << 8) | ((val * 10) << 16);
    return color;
}


__device__ __host__ int get_pixel_iter(int x, int y, int w, int h, int max_iter){
/* Returns number of iterations needed to classify whether point (x,y) is inside Mandelbrot set
@params - see dynamic_mandelbrot_k()
*/
    double a, b;
    Complex c1, c2;
    a = ((double)x - ((double)w / 2.0)) * 4.0 / (double)w;
    b = ((double)y - ((double)h / 2.0)) * 4.0 / (double)w;
    c1 = {0.0, 0.0};
    c2 = {a, b};
    int iter = 0;
    while (magnitude_squared(c1) < 4 && ++iter < max_iter){
        c1 = multiply_complex(c1, c1);
        c1.re += c2.re;
        c1.im += c2.im;
    }
    return iter;
}


__global__ void fill_rectangle_k(int iter, int x0, int y0, int w, int div_w, int div_h, int max_iter, int* results){
/* Fill array for rectangle positioned at x0, y0 (top left corner) of size div_w and div_h with color value based on iter
@params - see dynamic_mandelbrot_k()
*/
    int x = threadIdx.x + blockIdx.x * blockDim.x + x0;
    int y = threadIdx.y + blockIdx.y * blockDim.y + y0;
    if(x <= x0 + div_w && y <= y0 + div_h){
        if(iter < max_iter) results[y * w + x] = get_color_value(iter); //16777215
        else results[y * w + x] = 0;
    }
}


__device__ void fill_min_max_arrays(int pixels_per_thread, int pixels_in_border,
                                   int* min, int* max, int max_iter, int x0, int y0,
                                   int w, int h, int div_w, int div_h){
/* Fills shared memory arrays (min, max) with min and max values of iter in each thread's range.
When range (pixels_per_thread) <=1 it's just iter value in both arrays
@params min, max - shared memory arrays to be filled with min and max values of each thread range
@params - see dynamic_mandelbrot_k()
*/
    int index = blockDim.x * threadIdx.y + threadIdx.x;
    int start = index * pixels_per_thread;
    if(index < pixels_in_border){
        min[index] = max_iter;
        max[index] = 1;
    }
    else{
        min[index] = 1;
        max[index] = max_iter;
    }
    int iter;
    for(int i = start; i < start + pixels_per_thread; i++){
        if (i >= pixels_in_border) break;
        if (i < div_w){ // top
            iter = get_pixel_iter(x0 + i, y0, w, h, max_iter);
            if(iter <= min[index]) min[index] = iter;
            if(iter >= max[index]) max[index] = iter;
        }
        else if (i >= div_w  && i < 2 * div_w){ // bottom
            iter = get_pixel_iter(x0 + i - div_w, y0 + div_h - 1, w, h, max_iter);
            if(iter <= min[index]) min[index] = iter;
            if(iter >= max[index]) max[index] = iter;
        }
        else if (i >= 2 * div_w && (i - 2 * div_w) < div_h){ // left
            iter = get_pixel_iter(x0, y0 + (i - (2 * div_w)), w, h, max_iter);
            if(iter <= min[index]) min[index] = iter;
            if(iter >= max[index]) max[index] = iter;
        }
        else{ // right
            iter = get_pixel_iter(x0 + div_w - 1, y0 + (i - (2 * div_w) - div_h), w, h, max_iter);
            if(iter <= min[index]) min[index] = iter;
            if(iter >= max[index]) max[index] = iter;
        }
    }
}


__global__ void calculate_fill_rectangle_k(int x0, int y0, int w, int h, int div_w, int div_h, int max_iter, int* results){
/* Calculates values for rectangle originated at point (x0, y0) and size div_w, div_h and fills result array with them.
@params - see dynamic_mandelbrot_k()
*/
    int x = threadIdx.x + blockIdx.x * blockDim.x + x0;
    int y = threadIdx.y + blockIdx.y * blockDim.y + y0;
    int iter = get_pixel_iter(x, y, w, h, max_iter);
    if(x <= x0 + div_w && y <= y0 + div_h){
        if(iter < max_iter) results[y * w + x] = get_color_value(iter);
        else results[y * w + x] = 0;
    }
}

__device__ void reduce_min_max(int *min, int* max, int range){
/* Parallel reduction with finding min and max for min and max shared memory arrays
 @params min, max - shared memory arrays containing min and max iter values of each thread range
 @param range - number of items in arrays
 */
    int index = blockDim.x * threadIdx.y + threadIdx.x;
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>=1){
        if (index < stride && index + stride < range){
            min[index] = min[index + stride] < min[index] ? min[index + stride] : min[index];
            max[index] = max[index + stride] > max[index] ? max[index + stride] : max[index];
        }
        __syncthreads();
    }

}


__global__ void dynamic_mandelbrot_k(int x, int y, int w, int h,
                                     int max_iter, int *results,
                                     int depth, int div_w, int div_h,
                                     int pixels_per_thread, int pixels_in_border){
/* Main kernel, computes values of boundary pixels, checks whether they're all the same and dispatches actions based on that
 @params x, y - coordinates of origin point of rectangle to be computed by this block of threads
 @params w, h - width and height of computed image
 @param max_iter - iteration limit for Mandelbrot set computations
 @param results - results array to be copied back to host
 @param depth - recursive calls depth
 @params div_w, div_h - width and height of rectangle to be computed by this block of threads
 @param pixels_per_thread - number of border pixels which values are to be computed by each thread in block
 @param pixels_in_border - number of pixels in border of rectangle to be computed by this block of threads
 */
    x += div_w * blockIdx.x;
    y += div_h * blockIdx.y;

    __shared__ int min[BLOCK_SIZE_Y * BLOCK_SIZE_X];
    __shared__ int max[BLOCK_SIZE_Y * BLOCK_SIZE_X];
    fill_min_max_arrays(pixels_per_thread, pixels_in_border, min, max,
                       max_iter, x, y, w, h, div_w, div_h);
    __syncthreads();
    int range = MIN(pixels_in_border, BLOCK_SIZE_Y * BLOCK_SIZE_X);
    reduce_min_max(min, max, range);

    if (threadIdx.x == 0 && threadIdx.y == 0) { // flow is controlled by first thread of each block
        // if all boundary pixels are the same then simply fill whole rectangle with that value
        if(min[0] == max[0]){ // if minimum iter value equals to maximum, then all values are the same
            int iter = get_pixel_iter(x, y, w, h, max_iter);
            dim3 grid = {div_round_up(div_w, BLOCK_SIZE_X), div_round_up(div_h, BLOCK_SIZE_Y)};
            dim3 block = {BLOCK_SIZE_X, BLOCK_SIZE_Y};
            fill_rectangle_k<<<grid, block>>>(iter, x, y, w, div_w, div_h, max_iter, results);
        }
        // if not too fragmented, call self recursively
        else if(depth + 1 < MAX_RECURSION_DEPTH &&
            ((div_w / BLOCK_DIVISION) * (div_h / BLOCK_DIVISION)) > MIN_SIZE){
            dim3 grid = {BLOCK_DIVISION, BLOCK_DIVISION, 1};
            dim3 block = {BLOCK_SIZE_X, BLOCK_SIZE_Y, 1};
            int next_div_w = div_round_up(div_w, BLOCK_DIVISION);
            int next_div_h = div_round_up(div_h, BLOCK_DIVISION);
            int border_pixels = 2 * (next_div_w - 1) + 2 * (next_div_h - 1);
            int ppt = div_round_up(border_pixels, BLOCK_SIZE_Y * BLOCK_SIZE_X);
            dynamic_mandelbrot_k<<<grid, block>>>(x, y, w, h, max_iter,
                                                  results, depth +1,
                                                  next_div_w, next_div_h,
                                                  ppt, border_pixels);
        }
        else{ // else just fill it like normal (on pixel per thread basis)
            dim3 grid = {div_round_up(div_w, BLOCK_SIZE_X), div_round_up(div_h, BLOCK_SIZE_Y)};
            dim3 block = {BLOCK_SIZE_X, BLOCK_SIZE_Y};
            calculate_fill_rectangle_k<<<grid, block>>>(x, y, w, h, div_w, div_h, max_iter, results);
        }
    }
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

    dim3 grid = {INITIAL_SQUARES, INITIAL_SQUARES, 1};
    dim3 block = {BLOCK_SIZE_X, BLOCK_SIZE_Y, 1};

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

    int threads_in_block = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    int pixels_in_border = (2 * div_round_up(width - 1, INITIAL_SQUARES))
                         + (2 * div_round_up(height - 1, INITIAL_SQUARES));
    int pixels_per_thread = div_round_up(pixels_in_border, threads_in_block);
    dynamic_mandelbrot_k<<<grid, block>>>(0, 0, width, height,
                                          max_iter, device_result, 1,
                                          div_round_up(width, INITIAL_SQUARES),
                                          div_round_up(height, INITIAL_SQUARES),
                                          pixels_per_thread, pixels_in_border);
    cudaDeviceSynchronize();

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
