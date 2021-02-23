#include <stdio.h>
#include <string.h>

#include "opencl.h"

/* Builds a program executable from the program binary. */
int inclBuildProgram(cl_program program) {
	cl_int errcode_ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clBuildProgram %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Returns a message related to the error code. */
const char *inclCheckErrorCode(cl_int errcode) {
	switch (errcode) {
		case -1:
			return "CL_DEVICE_NOT_FOUND";
		case -2:
			return "CL_DEVICE_NOT_AVAILABLE";
		case -3:
			return "CL_COMPILER_NOT_AVAILABLE";
		case -4:
			return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5:
			return "CL_OUT_OF_RESOURCES";
		case -6:
			return "CL_OUT_OF_HOST_MEMORY";
		case -7:
			return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8:
			return "CL_MEM_COPY_OVERLAP";
		case -9:
			return "CL_IMAGE_FORMAT_MISMATCH";
		case -10:
			return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11:
			return "CL_BUILD_PROGRAM_FAILURE";
		case -12:
			return "CL_MAP_FAILURE";
		case -13:
			return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14:
			return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15:
			return "CL_COMPILE_PROGRAM_FAILURE";
		case -16:
			return "CL_LINKER_NOT_AVAILABLE";
		case -17:
			return "CL_LINK_PROGRAM_FAILURE";
		case -18:
			return "CL_DEVICE_PARTITION_FAILED";
		case -19:
			return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
		case -30:
			return "CL_INVALID_VALUE";
		case -31:
			return "CL_INVALID_DEVICE_TYPE";
		case -32:
			return "CL_INVALID_PLATFORM";
		case -33:
			return "CL_INVALID_DEVICE";
		case -34:
			return "CL_INVALID_CONTEXT";
		case -35:
			return "CL_INVALID_QUEUE_PROPERTIES";
		case -36:
			return "CL_INVALID_COMMAND_QUEUE";
		case -37:
			return "CL_INVALID_HOST_PTR";
		case -38:
			return "CL_INVALID_MEM_OBJECT";
		case -39:
			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40:
			return "CL_INVALID_IMAGE_SIZE";
		case -41:
			return "CL_INVALID_SAMPLER";
		case -42:
			return "CL_INVALID_BINARY";
		case -43:
			return "CL_INVALID_BUILD_OPTIONS";
		case -44:
			return "CL_INVALID_PROGRAM";
		case -45:
			return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46:
			return "CL_INVALID_KERNEL_NAME";
		case -47:
			return "CL_INVALID_KERNEL_DEFINITION";
		case -48:
			return "CL_INVALID_KERNEL";
		case -49:
			return "CL_INVALID_ARG_INDEX";
		case -50:
			return "CL_INVALID_ARG_VALUE";
		case -51:
			return "CL_INVALID_ARG_SIZE";
		case -52:
			return "CL_INVALID_KERNEL_ARGS";
		case -53:
			return "CL_INVALID_WORK_DIMENSION";
		case -54:
			return "CL_INVALID_WORK_GROUP_SIZE";
		case -55:
			return "CL_INVALID_WORK_ITEM_SIZE";
		case -56:
			return "CL_INVALID_GLOBAL_OFFSET";
		case -57:
			return "CL_INVALID_EVENT_WAIT_LIST";
		case -58:
			return "CL_INVALID_EVENT";
		case -59:
			return "CL_INVALID_OPERATION";
		case -60:
			return "CL_INVALID_GL_OBJECT";
		case -61:
			return "CL_INVALID_BUFFER_SIZE";
		case -62:
			return "CL_INVALID_MIP_LEVEL";
		case -63:
			return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64:
			return "CL_INVALID_PROPERTY";
		case -65:
			return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66:
			return "CL_INVALID_COMPILER_OPTIONS";
		case -67:
			return "CL_INVALID_LINKER_OPTIONS";
		case -68:
			return "CL_INVALID_DEVICE_PARTITION_COUNT";
		case -69:
			return "CL_INVALID_PIPE_SIZE";
		case -70:
			return "CL_INVALID_DEVICE_QUEUE";
		default:
			return "CL_INVALID_ERROR_CODE";
	}
}

/* Creates a buffer object. */
cl_mem inclCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr) {
	cl_int errcode_ret;
	cl_mem mem = clCreateBuffer(context, flags, size, host_ptr, &errcode_ret);
	if (errcode_ret != CL_SUCCESS || !mem) {
		fprintf(stderr, "Error: clCreateBuffer %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return NULL;
	} else {
		return mem;
	}
}

/* Create a command-queue on a specific device. */
cl_command_queue inclCreateCommandQueue(cl_context context, cl_device_id device) {
	cl_int errcode_ret;
	cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &errcode_ret);
	if (errcode_ret != CL_SUCCESS || !command_queue) {
		fprintf(stderr, "Error: clCreateCommandQueue %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return NULL;
	} else {
		return command_queue;
	}
}

/* Creates an OpenCL context. */
cl_context inclCreateContext(cl_device_id device) {
	cl_int errcode_ret;
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &errcode_ret);
	if (errcode_ret != CL_SUCCESS || !context) {
		fprintf(stderr, "Error: clCreateContext %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return NULL;
	} else {
		return context;
	}
}

/* Creates a kernel object. */
cl_kernel inclCreateKernel(cl_program program, const char *kernel_name) {
	cl_int errcode_ret;
	cl_kernel kernel = clCreateKernel(program, kernel_name, &errcode_ret);
	if (errcode_ret != CL_SUCCESS || !kernel) {
		fprintf(stderr, "Error: clCreateKernel %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return NULL;
	} else {
		return kernel;
	}
}

/* Creates a program object for a context, and loads specified binary data into the program object. */
cl_program inclCreateProgramWithBinary(cl_context context, cl_device_id device, size_t length, const unsigned char *binary) {
	cl_int errcode_ret;
	cl_program program = clCreateProgramWithBinary(context, 1, &device, &length, &binary, NULL, &errcode_ret);
	if (errcode_ret != CL_SUCCESS || !program) {
		fprintf(stderr, "Error: clCreateProgramWithBinary %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return NULL;
	} else {
		return program;
	}
}

/* Enqueues a command to indicate which device a memory object should be associated with. */
int inclEnqueueMigrateMemObject(cl_command_queue command_queue, cl_mem mem_object, cl_mem_migration_flags flags) {
	cl_int errcode_ret = clEnqueueMigrateMemObjects(command_queue, 1, &mem_object, flags, 0, NULL, NULL);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clEnqueueMigrateMemObjects %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Enqueue commands to read from a buffer object to host memory. */
int inclEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, size_t offset, size_t cb, void *ptr) {
	cl_int errcode_ret = clEnqueueReadBuffer(command_queue, buffer, CL_FALSE, offset, cb, ptr, 0, NULL, NULL);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clEnqueueReadBuffer %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Enqueues a command to execute a kernel on a device. */
int inclEnqueueTask(cl_command_queue command_queue, cl_kernel kernel) {
	cl_int errcode_ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clEnqueueTask %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Enqueue commands to write to a buffer object from host memory. */
int inclEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, size_t offset, size_t cb, const void *ptr) {
	cl_int errcode_ret = clEnqueueWriteBuffer(command_queue, buffer, CL_FALSE, offset, cb, ptr, 0, NULL, NULL);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clEnqueueWriteBuffer %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Blocks until all previously queued OpenCL commands in a command-queue are issued to the associated device and have completed. */
int inclFinish(cl_command_queue command_queue) {
	cl_int errcode_ret = clFinish(command_queue);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clFinish %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Obtain specified device, if available. */
cl_device_id inclGetDeviceID(cl_platform_id platform, cl_uint device_id) {
	cl_device_id device = (cl_device_id) malloc(sizeof(cl_device_id));
	if (!device) {
		fprintf(stderr, "Error: malloc\n");
		return NULL;
	}

	cl_uint num_devices;
	inclGetDeviceIDs(platform, 0, NULL, &num_devices);

	cl_device_id *devices = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
	if (!devices) {
		free(device);
		fprintf(stderr, "Error: malloc\n");
		return NULL;
	}

	inclGetDeviceIDs(platform, num_devices, devices, NULL);

	cl_uint i;
	for (i = 0; i < num_devices; i++) {
		if (i == device_id) {
			device = devices[i];
			break;
		}
	}

	free(devices);

	if (i == num_devices) {
		free(device);
		fprintf(stderr, "Error: clGetDeviceID\n");
		return NULL;
	}

	return device;
}

/* Obtain the list of devices available on a platform. */
int inclGetDeviceIDs(cl_platform_id platform, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices) {
	cl_int errcode_ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_entries, devices, num_devices);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clGetDeviceIDs %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Get specific information about the OpenCL device. */
int inclGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
	cl_int errcode_ret = clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clGetDeviceInfo %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Get specific information about the OpenCL kernel. */
int inclGetKernelInfo(cl_kernel kernel, cl_kernel_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
	cl_int errcode_ret = clGetKernelInfo(kernel, param_name, param_value_size, param_value, param_value_size_ret);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clGetKernelInfo %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Get specific information about the OpenCL buffer. */
int inclGetMemObjectInfo(cl_mem memobj, cl_mem_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret ) {
	cl_int errcode_ret = clGetMemObjectInfo(memobj, param_name, param_value_size, param_value, param_value_size_ret);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clGetMemObjectInfo %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Obtain specified platform, if available. */
cl_platform_id inclGetPlatformID(const char *platform_id) {
	cl_platform_id platform = (cl_platform_id) malloc(sizeof(cl_platform_id));
	if (!platform) {
		fprintf(stderr, "Error: malloc\n");
		return NULL;
	}

	cl_uint num_platforms;
	inclGetPlatformIDs(0, NULL, &num_platforms);

	cl_platform_id *platforms = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
	if (!platforms) {
		free(platform);
		fprintf(stderr, "Error: malloc\n");
		return NULL;
	}

	inclGetPlatformIDs(num_platforms, platforms, NULL);

	cl_uint i;
	for (i = 0; i < num_platforms; i++) {
		size_t platform_name_size;
		inclGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &platform_name_size);

		char *platform_name = (char *) malloc(platform_name_size * sizeof(char));
		if (!platform_name) {
			free(platform);
			free(platforms);
			fprintf(stderr, "Error: malloc\n");
			return NULL;
		}

		inclGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_size, platform_name, NULL);

		if (strstr(platform_name, platform_id)) {
			free(platform_name);

			platform = platforms[i];
			break;
		}

		free(platform_name);
	}

	free(platforms);

	if (i == num_platforms) {
		free(platform);
		fprintf(stderr, "Error: clGetPlatformID\n");
		return NULL;
	}

	return platform;
}

/* Obtain the list of platforms available. */
int inclGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms) {
	cl_int errcode_ret = clGetPlatformIDs(num_entries, platforms, num_platforms);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clGetPlatformIDs %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Get specific information about the OpenCL platform. */
int inclGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
	cl_int errcode_ret = clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clGetPlatformInfo %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Decrements the command_queue reference count. */
int inclReleaseCommandQueue(cl_command_queue command_queue) {
	cl_int errcode_ret = clReleaseCommandQueue(command_queue);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clReleaseCommandQueue %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Decrement the context reference count. */
int inclReleaseContext(cl_context context) {
	cl_int errcode_ret = clReleaseContext(context);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clReleaseContext %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Decrements the kernel reference count. */
int inclReleaseKernel(cl_kernel kernel) {
	cl_int errcode_ret = clReleaseKernel(kernel);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clReleaseKernel %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Decrements the memory object reference count. */
int inclReleaseMemObject(cl_mem memobj) {
	cl_int errcode_ret = clReleaseMemObject(memobj);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clReleaseMemObject %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Decrements the program reference count. */
int inclReleaseProgram(cl_program program) {
	cl_int errcode_ret = clReleaseProgram(program);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clReleaseProgram %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}

/* Increments the memory object reference count. */
int inclRetainMemObject(cl_mem memobj) {
		cl_int errcode_ret = clRetainMemObject(memobj);
		if (errcode_ret != CL_SUCCESS) {
			fprintf(stderr, "Error: clRetainMemObject %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
			return EXIT_FAILURE;
		} else {
			return EXIT_SUCCESS;
		}
}

/* Used to set the argument value for a specific argument of a kernel. */
int inclSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
	cl_int errcode_ret = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
	if (errcode_ret != CL_SUCCESS) {
		fprintf(stderr, "Error: clSetKernelArg %s (%d)\n", inclCheckErrorCode(errcode_ret), errcode_ret);
		return EXIT_FAILURE;
	} else {
		return EXIT_SUCCESS;
	}
}
