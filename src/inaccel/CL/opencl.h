#ifndef OPENCL_H
#define OPENCL_H

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.h>

/* Builds a program executable from the program binary. */
int inclBuildProgram(cl_program program);

/* Returns a message related to the error code. */
const char *inclCheckErrorCode(cl_int errcode);

/* Creates a buffer object. */
cl_mem inclCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr);

/* Create a command-queue on a specific device. */
cl_command_queue inclCreateCommandQueue(cl_context context, cl_device_id device);

/* Creates an OpenCL context. */
cl_context inclCreateContext(const cl_device_id device);

/* Creates a kernel object. */
cl_kernel inclCreateKernel(cl_program program, const char *kernel_name);

/* Creates a program object for a context, and loads specified binary data into the program object. */
cl_program inclCreateProgramWithBinary(cl_context context, cl_device_id device, size_t length, const unsigned char *binary);

/* Enqueues a command to indicate which device a memory object should be associated with. */
int inclEnqueueMigrateMemObject(cl_command_queue command_queue, cl_mem memobj, cl_mem_migration_flags flags);

/* Enqueue commands to read from a buffer object to host memory. */
int inclEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, size_t offset, size_t cb, void *ptr);

/* Enqueues a command to execute a kernel on a device. */
int inclEnqueueTask(cl_command_queue command_queue, cl_kernel kernel);

/* Enqueue commands to write to a buffer object from host memory. */
int inclEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, size_t offset, size_t cb, const void *ptr);

/* Blocks until all previously queued OpenCL commands in a command-queue are issued to the associated device and have completed. */
int inclFinish(cl_command_queue command_queue);

/* Obtain specified device, if available. */
cl_device_id inclGetDeviceID(cl_platform_id platform, cl_uint device_id);

/* Obtain the list of devices available on a platform. */
int inclGetDeviceIDs(cl_platform_id platform, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices);

/* Get specific information about the OpenCL device. */
int inclGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

/* Get specific information about the OpenCL kernel. */
int inclGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

/* Get specific information about the OpenCL buffer. */
int inclGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret );

/* Obtain specified  platform, if available. */
cl_platform_id inclGetPlatformID(const char *platform_id);

/* Obtain the list of platforms available. */
int inclGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms);

/* Get specific information about the OpenCL platform. */
int inclGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret);

/* Decrements the command_queue reference count. */
int inclReleaseCommandQueue(cl_command_queue command_queue);

/* Decrement the context reference count. */
int inclReleaseContext(cl_context context);

/* Decrements the kernel reference count. */
int inclReleaseKernel(cl_kernel kernel);

/* Decrements the memory object reference count. */
int inclReleaseMemObject(cl_mem memobj);

/* Decrements the program reference count. */
int inclReleaseProgram(cl_program program);

/* Increments the memory object reference count. */
int inclRetainMemObject(cl_mem memobj);

/* Used to set the argument value for a specific argument of a kernel. */
int inclSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value);

#endif
