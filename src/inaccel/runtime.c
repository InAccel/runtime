#include <glob.h>
#include <libgen.h>
#include <limits.h>
#include <regex.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "CL/opencl.h"
#include "runtime.h"

#ifdef Xilinx
struct mem_data {
	uint8_t m_type;          // enum corresponding to mem_type.
	uint8_t m_used;          // if 0 this bank is not present
	uint8_t padding[6];      // 8 Byte alignment padding (initialized to zeros)
	union {
		uint64_t m_size;     // if mem_type DDR, then size in KB;
		uint64_t route_id;   // if streaming then "route_id"
	};
	union {
		uint64_t m_base_address; // if DDR then the base address;
		uint64_t flow_id;        // if streaming then "flow id"
	};
	unsigned char m_tag[16]; // DDR: BANK0,1,2,3, has to be null terminated; if streaming then stream0, 1 etc
};

struct mem_topology {
	int32_t m_count; //Number of mem_data
	struct mem_data m_mem_data[1]; //Should be sorted on mem_type
};

typedef struct{
	unsigned flags;
	void *obj;
	void *param;
} cl_mem_ext_ptr_t;
#endif

/* CL buffer struct (Type). */
struct _cl_buffer {
#ifdef Intel
	void *ptr;
	size_t size;
#endif
	cl_command_queue command_queue;
	cl_mem mem;
	cl_memory memory;
};

/* CL compute unit struct (Type). */
struct _cl_compute_unit {
	cl_command_queue command_queue;
	cl_kernel kernel;
	cl_mem *args;
};

/* CL memory struct (Type). */
struct _cl_memory {
	unsigned int id;
	cl_resource resource;
	cl_mem page_buf;
};

/* CL resource struct (Type). */
struct _cl_resource {
	// Virtual device variables
	cl_context context;
	cl_device_id device_id;
	cl_platform_id platform_id;
	cl_program program;

	// Physical device variables
	char *name;
	char *root_path;
	char *vendor;
	char *version;

	char *temperature;
#ifdef Intel
	char *sdr;
	char *sensors;
	size_t mem_capacity;
#endif
#ifdef Xilinx
	char *power;
	char *serial_no;
	struct mem_topology *topology;
#endif
};

#ifdef Intel
#define SIGN_EXT(val, bitpos) (((val) ^ (1 << (bitpos))) - (1 << (bitpos)))

float intel_get_power(cl_resource resource) {
	unsigned char calc_params[6];
	unsigned char reading;

	if (resource->sdr && resource->sensors) {
		FILE *sdr_file = fopen(resource->sdr, "r");
		FILE *sensors_file = fopen(resource->sensors, "r");

		if (!sdr_file || !sensors_file)
			return -1.0f;

		if (fseek(sdr_file, 24, SEEK_SET))
			return -1.0f;

		int ret;
		if(!(ret = fread(calc_params, sizeof(char), 6, sdr_file)))
			return -1.0f;

		fclose(sdr_file);

		if (fseek(sensors_file, 4, SEEK_SET))
			return -1.0f;


		if(!(ret = fread(&reading, sizeof(char), 1, sensors_file)))
			return -1.0f;

		fclose(sensors_file);

		int32_t B_val = ((calc_params[3] >> 6 & 0x3) << 8) | calc_params[2];
		B_val = SIGN_EXT(B_val, 9);

		int32_t M_val = ((calc_params[1] >> 6 & 0x3) << 8) | calc_params[0];
		M_val = SIGN_EXT(M_val, 9);

		int32_t R_exp = calc_params[5] >> 4 & 0x0F;
		R_exp = SIGN_EXT(R_exp, 3);

		int32_t B_exp = calc_params[5] & 0x0F;
		B_exp = SIGN_EXT(B_exp, 3);

		double M = M_val;
		double B = B_val;

		int i;
		if (B_exp >= 0) {
			for (i = 0; i < B_exp; i++) {
				B *= 10.0;
			}
		} else {
			for (i = B_exp; i; i++) {
				B /= 10.0;
			}
		}

		double sensor_value = M * reading + B;

		if (R_exp >= 0) {
			for (i = 0; i < R_exp; i++) {
				sensor_value *= 10.0;
			}
		} else {
			for (i = R_exp; i; i++) {
				sensor_value /= 10.0;
			}
		}

		return sensor_value;
	}

	return -1.0f;
}
#endif

/**
 * Waits on the host thread until all previous copy commands are issued to the associated resource and have completed.
 * @param buffer Refers to a valid buffer object.
 * @return 0 on success; 1 on failure.
 */
int await_buffer_copy(cl_buffer buffer) {
	return inclFinish(buffer->command_queue);
}

/**
 * Waits on the host thread until all previous run commands are issued to the associated resource and have completed.
 * @param buffer Refers to a valid compute unit object.
 * @return 0 on success; 1 on failure.
 */
int await_compute_unit_run(cl_compute_unit compute_unit) {
	return inclFinish(compute_unit->command_queue);
}

/**
 * Commands to read from a buffer object to host memory.
 * @param buffer Refers to a valid buffer object.
 * @return 0 on success; 1 on failure.
 */
int copy_from_buffer(cl_buffer buffer) {
#if Intel
	return inclEnqueueReadBuffer(buffer->command_queue, buffer->mem, 0, buffer->size, buffer->ptr);
#endif
#if Xilinx
	return inclEnqueueMigrateMemObject(buffer->command_queue, buffer->mem, 1);
#endif
}

/**
 * Commands to write to a buffer object from host memory.
 * @param buffer Refers to a valid buffer object.
 * @return 0 on success; 1 on failure.
 */
int copy_to_buffer(cl_buffer buffer) {
#ifdef Intel
	return inclEnqueueWriteBuffer(buffer->command_queue, buffer->mem, 0, buffer->size, buffer->ptr);
#endif
#ifdef Xilinx
	return inclEnqueueMigrateMemObject(buffer->command_queue, buffer->mem, 0);
#endif
}

/**
 * Creates a buffer object.
 * @param memory A valid memory used to create the buffer object.
 * @param size The size in bytes of the buffer memory object to be allocated.
 * @param array A pointer to the buffer data that should already be allocated by the application. The size of the buffer that address points to must be greater than or equal to the size bytes.
 * @return The buffer.
 */
cl_buffer create_buffer(cl_memory memory, size_t size, void *array) {
	cl_buffer buffer = (cl_buffer) calloc(1, sizeof(struct _cl_buffer));

#ifdef Intel
	buffer->size = size;
	buffer->ptr = array;

	cl_uint CL_MEMORY = memory->id << 16;

	if (!(buffer->mem = inclCreateBuffer(memory->resource->context, CL_MEM_READ_WRITE | CL_MEMORY, size, NULL))) goto CATCH;
#endif
#ifdef Xilinx
	cl_uint CL_MEM_EXT_PTR = 1 << 31;
	cl_uint CL_MEMORY = memory->id | (1 << 31);

	cl_mem_ext_ptr_t ext_ptr;
	ext_ptr.flags = CL_MEMORY;
	ext_ptr.obj = array;
	ext_ptr.param = 0;

	if (!(buffer->mem = inclCreateBuffer(memory->resource->context, CL_MEM_EXT_PTR | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size, &ext_ptr))) goto CATCH;
#endif

	if (!(buffer->command_queue = inclCreateCommandQueue(memory->resource->context, memory->resource->device_id))) goto CATCH;

	buffer->memory = memory;

	return buffer;
CATCH:
	free(buffer);

	return NULL;
}

/**
 * Creates a compute unit object.
 * @param resource A valid resource used to create the compute unit object.
 * @param name A function name in the binary executable.
 * @return The compute unit.
 */
cl_compute_unit create_compute_unit(cl_resource resource, const char *name) {
	cl_compute_unit compute_unit = (cl_compute_unit) calloc(1, sizeof(struct _cl_compute_unit));

	if (!(compute_unit->kernel = inclCreateKernel(resource->program, name))) goto CATCH;

	if (!(compute_unit->command_queue = inclCreateCommandQueue(resource->context, resource->device_id))) goto CATCH;

	cl_uint num_arguments;
	inclGetKernelInfo(compute_unit->kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &num_arguments, NULL);

	compute_unit->args = (cl_mem *) calloc(num_arguments, sizeof(cl_mem));

	return compute_unit;
CATCH:
	free(compute_unit);

	return NULL;
}

/**
 * Creates a memory object.
 * @param resource A valid resource used to create the memory object.
 * @param index The index associated with this memory.
 * @return The memory.
 */
cl_memory create_memory(cl_resource resource, unsigned int index) {
#ifdef Intel
	// Intel has only one memory (for now)
	if (!index) {
#endif
#ifdef Xilinx
	if (resource->topology && (index < resource->topology->m_count)) {
#endif
		cl_memory memory = (cl_memory) calloc(1, sizeof(struct _cl_memory));

		memory->id = index;
		memory->resource = resource;

		cl_uint CL_MEM_EXT_PTR = 1 << 31;
		cl_uint CL_MEMORY = memory->id | (1 << 31);

		cl_mem_ext_ptr_t ext_ptr;
		ext_ptr.flags = CL_MEMORY;
		ext_ptr.obj = NULL;
		ext_ptr.param = 0;

		if (!(memory->page_buf = inclCreateBuffer(memory->resource->context, CL_MEM_EXT_PTR | CL_MEM_WRITE_ONLY, 4096, &ext_ptr))) return NULL;

		return memory;
	}

	return NULL;
}

/**
 * Creates a resource object.
 * @param device_id The device associated with this resource.
 * @return The resource.
 */
cl_resource create_resource(unsigned int device_id) {
	cl_resource resource = (cl_resource) calloc(1, sizeof(struct _cl_resource));

#ifdef Intel
	resource->vendor = strdup("intel");
	if (!(resource->platform_id = inclGetPlatformID("Intel"))) goto CATCH;

	if (!(resource->device_id = inclGetDeviceID(resource->platform_id, device_id))) goto CATCH;

	if (!(resource->context = inclCreateContext(resource->device_id))) goto CATCH;

	cl_ulong mem_capacity;
	inclGetDeviceInfo(resource->device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_capacity), &mem_capacity, NULL);
	// mem_capacity is returned into MiB, so we convert it to bytes
	resource->mem_capacity = mem_capacity;

	size_t raw_name_size;
	inclGetDeviceInfo(resource->device_id, CL_DEVICE_NAME, 0, NULL, &raw_name_size);

	char *raw_name = (char *) malloc(raw_name_size * sizeof(char));
	if (!raw_name) {
		fprintf(stderr, "Error: malloc\n");
		return resource;
	}

	memset(raw_name, 0, raw_name_size * sizeof(char));

	inclGetDeviceInfo(resource->device_id, CL_DEVICE_NAME, raw_name_size, raw_name, NULL);

	char *name = (char *) malloc(raw_name_size * sizeof(char));
	if (!name) {
		free(raw_name);
		fprintf(stderr, "Error: malloc\n");
		return resource;
	}

	memset(name, 0, raw_name_size * sizeof(char));

	const char *regex = "^([^ : ]+) : .*_(.+)0000(.).$";

	regex_t compile;
	regmatch_t group[4];

	if (regcomp(&compile, regex, REG_EXTENDED)) {
		free(raw_name);
		free(name);
		fprintf(stderr, "Error: regcomp\n");
		return resource;
	}

	long int major, minor;

	if (!regexec(&compile, raw_name, 4, group, 0)) {
		strncpy(name, raw_name + group[1].rm_so, group[1].rm_eo - group[1].rm_so);
		resource->name = name;

		char *tmp;

		tmp = strndup(raw_name + group[2].rm_so, group[2].rm_eo - group[2].rm_so);
		if (!tmp) {
			free(raw_name);
			fprintf(stderr, "Error: strndup\n");
			return resource;
		}
		major = strtol(tmp, NULL, 16);
		free(tmp);

		tmp = strndup(raw_name + group[3].rm_so, group[3].rm_eo - group[3].rm_so);
		if (!tmp) {
			free(raw_name);
			fprintf(stderr, "Error: strndup\n");
			return resource;
		}
		minor = strtol(tmp, NULL, 16);
		free(tmp);
	} else {
		free(raw_name);
		fprintf(stderr, "Error: regexec\n");
		return resource;
	}

	regfree(&compile);

	free(raw_name);

	size_t UUID = 32;

	char *version = (char *) malloc((UUID + 1) * sizeof(char));
	if (!version) {
		fprintf(stderr, "Error: malloc\n");
		return resource;
	}

	memset(version, 0, (UUID + 1) * sizeof(char));

	glob_t dev;
	if (!glob("/sys/devices/pci*/*/{,/*}/fpga/intel-fpga-dev.*/intel-fpga-port.*/dev", GLOB_NOSORT | GLOB_BRACE, NULL, &dev)) {
		size_t i;
		for (i = 0; i < dev.gl_pathc; i++) {
			long int dev_major;
			long int dev_minor;

			FILE *dev_file = fopen(dev.gl_pathv[i], "r");
			if (!dev_file) {
				continue;
			}
			if (fscanf(dev_file, "%ld:%ld", &dev_major, &dev_minor) == EOF) {
				continue;
			}
			if (fclose(dev_file) == EOF) {
				continue;
			}

			if ((major == dev_major) && (minor == dev_minor)) {
				char path[PATH_MAX];
				sprintf(path, "%s/intel-fpga-fme.*/pr/interface_id", dirname(dirname(dev.gl_pathv[i])));

				glob_t interface_id;
				if (!glob(path, GLOB_NOSORT, NULL, &interface_id)) {
					FILE *interface_id_file = fopen(interface_id.gl_pathv[0], "r");
					if (!interface_id_file) {
						continue;
					}
					if (!fgets(version, UUID + 1, interface_id_file)) {
						continue;
					}
					if (fclose(interface_id_file) == EOF) {
						continue;
					}

					resource->root_path = strdup(dirname(dirname(interface_id.gl_pathv[0])));

					globfree(&interface_id);
				}

				break;
			}
		}
		globfree(&dev);
	}

	if (!strlen(version)) {
		free(version);
		fprintf(stderr, "Error: strlen\n");
		return resource;
	}
	resource->version = version;

	char path[PATH_MAX];
	sprintf(path, "%s/%s", resource->root_path, "thermal_mgmt/temperature");
	resource->temperature = strdup(path);

	sprintf(path, "%s/%s", resource->root_path, "avmmi-bmc.*.auto/bmc_info");

	glob_t bmc;
	if (!glob(path, GLOB_NOSORT, NULL, &bmc)) {
		sprintf(path, "%s/%s", bmc.gl_pathv[0], "sdr");
		resource->sdr = strdup(path);

		sprintf(path, "%s/%s", bmc.gl_pathv[0], "sensors");
		resource->sensors = strdup(path);

		globfree(&bmc);
	}
#endif
#ifdef Xilinx
	resource->vendor = strdup("xilinx");

	if (!(resource->platform_id = inclGetPlatformID("Xilinx"))) goto CATCH;

	if (!(resource->device_id = inclGetDeviceID(resource->platform_id, device_id))) goto CATCH;

	if (!(resource->context = inclCreateContext(resource->device_id))) goto CATCH;

	size_t raw_name_size;
	inclGetDeviceInfo(resource->device_id, CL_DEVICE_NAME, 0, NULL, &raw_name_size);

	char *raw_name = (char *) malloc(raw_name_size * sizeof(char));
	if (!raw_name) {
		fprintf(stderr, "Error: malloc\n");
		return resource;
	}

	memset(raw_name, 0, raw_name_size * sizeof(char));

	inclGetDeviceInfo(resource->device_id, CL_DEVICE_NAME, raw_name_size, raw_name, NULL);

	char *name = (char *) malloc(raw_name_size * sizeof(char));
	if (!name) {
		free(raw_name);
		fprintf(stderr, "Error: malloc\n");
		return resource;
	}

	memset(name, 0, raw_name_size * sizeof(char));

	char *version = (char *) malloc(raw_name_size * sizeof(char));
	if (!version) {
		free(raw_name);
		free(name);
		fprintf(stderr, "Error: malloc\n");
		return resource;
	}

	memset(version, 0, raw_name_size * sizeof(char));

	const char *regex = "^xilinx_([^_]+)_(.*)_([^_]+)_([^_]+)$";

	regex_t compile;
	regmatch_t group[5];

	if (regcomp(&compile, regex, REG_EXTENDED)) {
		free(raw_name);
		free(name);
		free(version);
		fprintf(stderr, "Error: regcomp\n");
		return resource;
	}

	if (!regexec(&compile, raw_name, 5, group, 0)) {
		strncpy(name, raw_name + group[1].rm_so, group[1].rm_eo - group[1].rm_so);

		strncpy(version, raw_name + group[2].rm_so, group[2].rm_eo - group[2].rm_so);
		strcat(version, "_");
		strncat(version, raw_name + group[3].rm_so, group[3].rm_eo - group[3].rm_so);
		strcat(version, ".");
		strncat(version, raw_name + group[4].rm_so, group[4].rm_eo - group[4].rm_so);
	} else {
		fprintf(stderr, "Error: regexec\n");
		return resource;
	}

	resource->name = name;
	resource->version = version;

	regfree(&compile);

	free(raw_name);

	glob_t dev;
	unsigned idx = 0;

	if (!glob("/sys/bus/pci/drivers/{xocl,xuser}/*:*:*.*", GLOB_BRACE, NULL, &dev)) {
		size_t i;
		for (i = dev.gl_pathc; i > 0; i--) {
			char temp_dev[PATH_MAX / 2] = {0};
			if (readlink(dev.gl_pathv[i - 1], temp_dev, sizeof(temp_dev) - 1) == 1)
				return resource;

			char path[PATH_MAX];
			sprintf(path, "%s/%s/ready", dirname(dev.gl_pathv[i - 1]), temp_dev);

			FILE *ready_file = fopen(path, "r");
			if (!ready_file) continue;

			unsigned ready;
			if(fscanf(ready_file, "0x%d", &ready) <= 0) continue;

			fclose(ready_file);

			if (ready) {
				if (idx == device_id) {
					resource->root_path = strdup(dirname(path));

					break;
				}
				else idx++;
			}
		}
	}

	globfree(&dev);

	char path[PATH_MAX];
	sprintf(path, "%s/xmc.*", resource->root_path);

	glob_t xmc;
	if (!glob(path, GLOB_NOSORT, NULL, &xmc)) {
		sprintf(path, "%s/%s", xmc.gl_pathv[0], "xmc_fpga_temp");
		resource->temperature = strdup(path);

		sprintf(path, "%s/%s", xmc.gl_pathv[0], "xmc_power");
		resource->power = strdup(path);


		sprintf(path, "%s/%s", xmc.gl_pathv[0], "serial_num");
		FILE *serial = fopen(path, "r");
		if (serial) {
			char serial_no[50];
			serial_no[0] = '\0';

			if (fscanf(serial,"%s", serial_no)) {
				resource->serial_no = strdup(serial_no);
			}

			fclose(serial);
		}
	}
	globfree(&xmc);

	sprintf(path, "%s/icap.*", resource->root_path);

	glob_t icap;
	if (!glob(path, GLOB_NOSORT, NULL, &icap)) {
		sprintf(path, "%s/%s", icap.gl_pathv[0], "mem_topology");

		FILE *mem_topology = fopen(path, "r");
		if (mem_topology) {
			//PATH_MAX bytes should be sufficient for mem_topology
			char *topology = (char *) malloc(PATH_MAX);
			size_t topology_bytes = fread(topology, sizeof(char), PATH_MAX, mem_topology);

			fclose(mem_topology);

			if (!topology_bytes)
				free(topology);
			else {
				resource->topology = (struct mem_topology *) topology;
			}
		}
	}
	globfree(&icap);
#endif
	return resource;
CATCH:
	free(resource);

	return NULL;
}

/**
 * Get the size of a memory.
 * @param memory Refers to a valid memory object.
 * @return The size of the memory in bytes
 */
size_t get_memory_size(cl_memory memory) {
#ifdef Intel
	return memory->resource->mem_capacity;
#endif
#ifdef Xilinx
	// Cross-check that the memory still exists in the topology
	if (memory->resource->topology->m_mem_data[memory->id].m_used) {
		return memory->resource->topology->m_mem_data[memory->id].m_size * 1024 - 4096; // minus page_buf size
	}

	return 0;
#endif
}


/**
* Get the type of a memory.
* @param memory Refers to a valid memory object.
* @return The memory type.
*/
char *get_memory_type(cl_memory memory) {
#ifdef Intel
	return strdup("DDR");
#endif
#ifdef Xilinx
	if (memory->resource->topology) {
		char *tag = (char *) malloc(strlen((const char *) memory->resource->topology->m_mem_data[memory->id].m_tag));
		strcpy(tag, (const char *) memory->resource->topology->m_mem_data[memory->id].m_tag);

		strtok(tag, "[");

		if (strcmp(tag, "bank0") && strcmp(tag, "bank1") && strcmp(tag, "bank2") && strcmp(tag, "bank3")) {
			return tag;
		} else{
			free(tag);
			return strdup("DDR");
		}

	} else {
		return strdup("-");
	}
#endif
}

/**
 * Get the name of a resource.
 * @param resource Refers to a valid resource object.
 * @return The name.
 */
char *get_resource_name(cl_resource resource) {
	if (resource->name)
		return strdup(resource->name);
	else
		return strdup("-");
}

/**
 * Get the power consumption of a resource.
 * @param resource Refers to a valid resource object.
 * @return The power consumed.
 */
float get_resource_power(cl_resource resource) {
#ifdef Intel
	return intel_get_power(resource);
#endif
#ifdef Xilinx
	if (resource->power) {
		FILE *power_file = fopen(resource->power, "r");

		if (power_file) {
			char tmp[10] = {0};

			int ret = fscanf(power_file, "%s", tmp);
			fclose(power_file);

			if(ret != EOF) {
				// Power is obtained in uWatts so we convert it to Watts
				float value = ((double) strtoul(tmp, NULL, 10)) / 1000000;

				// Keep only 2 decimals
				return value;
			}
		}
	}

	return -1.0f;
#endif
}

/**
 * Get the serial number of a resource.
 * @param resource Refers to a valid resource object.
 * @return The serial number.
 */
char *get_resource_serial_no(cl_resource resource) {
#ifdef Xilinx
	if(resource->serial_no)
		return strdup(resource->serial_no);
#endif
	return strdup("-");
}

/**
 * Get the current temperature of a resource.
 * @param resource Refers to a valid resource object.
 * @return The current temperature of the resource in degrees Celsius.
 */
float get_resource_temperature(cl_resource resource) {
	if (resource->temperature) {
		FILE *temp_file = fopen(resource->temperature, "r");

		if (temp_file) {
			char temperature[4] = {0};

			int ret = fscanf(temp_file, "%s", temperature);
			fclose(temp_file);

			if(ret != EOF) {
				return strtof(temperature, NULL);
			}
		}
	}

	return -1;
}

/**
 * Get the vendor of a resource.
 * @param resource Refers to a valid resource object.
 * @return The vendor.
 */
char *get_resource_vendor(cl_resource resource) {
	if (resource->vendor)
		return strdup(resource->vendor);
	else
		return strdup("-");
}

/**
 * Get the version (bsp/shell) of a resource.
 * @param resource Refers to a valid resource object.
 * @return The version.
 */
char *get_resource_version(cl_resource resource) {
	if (resource->version)
		return strdup(resource->version);
	else
		return strdup("-");
}

/**
 * Loads the specified binary executable bits into the resource object.
 * @param resource Refers to a valid resource object.
 * @param size The size in bytes of the binary to be loaded.
 * @param binary Pointer to binary to be loaded. The binary specified contains the bits that describe the executable that will be run on the associated resource.
 * @return 0 on success; 1 on failure.
 */
int program_resource_with_binary(cl_resource resource, size_t size, const void *binary) {
	printf("Programming device, size: %lu, binary: %p\n",size, binary);
	fflush(stdout);

	if (resource->program) if (inclReleaseProgram(resource->program)) goto CATCH;

	if (!(resource->program = inclCreateProgramWithBinary(resource->context, resource->device_id, size, binary))) goto CATCH;

	if (inclBuildProgram(resource->program)) goto CATCH;

#ifdef Intel
	return EXIT_SUCCESS;
#endif
// In Intel the mem_topology remains unchangeable
#ifdef Xilinx
	// mem_topology changes every time we program the device
	if (resource->topology) free(resource->topology);

	char path[PATH_MAX];
	sprintf(path, "%s/icap.*", resource->root_path);

	glob_t icap;
	if (!glob(path, GLOB_NOSORT, NULL, &icap)) {
		sprintf(path, "%s/%s", icap.gl_pathv[0], "mem_topology");
		globfree(&icap);

		FILE *mem_topology = fopen(path, "r");
		if (mem_topology) {
			//PATH_MAX bytes should be sufficient for mem_topology
			char *topology = (char *) malloc(PATH_MAX);
			size_t topology_bytes = fread(topology, sizeof(char), PATH_MAX, mem_topology);

			fclose(mem_topology);

			if (!topology_bytes) {
				free(topology);
				goto CATCH;
			}
			else {
				resource->topology = (struct mem_topology *) topology;
				return EXIT_SUCCESS;
			}
		}
	}
#endif

CATCH:
	return EXIT_FAILURE;
}

/**
 * Deletes a buffer object.
 * @param buffer Refers to a valid buffer object.
 */
void release_buffer(cl_buffer buffer) {
	inclReleaseCommandQueue(buffer->command_queue);

	inclReleaseMemObject(buffer->mem);

	free(buffer);
}

/**
 * Deletes a compute unit object.
 * @param compute_unit Refers to a valid compute unit object.
 */
void release_compute_unit(cl_compute_unit compute_unit) {
	inclReleaseCommandQueue(compute_unit->command_queue);

	inclReleaseKernel(compute_unit->kernel);

	free(compute_unit->args);

	free(compute_unit);
}

/**
 * Deletes a memory object.
 * @param memory Refers to a valid memory object.
 */
void release_memory(cl_memory memory) {
	inclReleaseMemObject(memory->page_buf);

	if(memory) free(memory);
}

/**
 * Deletes a resource object.
 * @param resource Refers to a valid resource object.
 */
void release_resource(cl_resource resource) {
	if (resource->program) inclReleaseProgram(resource->program);

	inclReleaseContext(resource->context);

	if (resource->name) free(resource->name);
	if (resource->root_path) free(resource->root_path);
	if (resource->vendor) free(resource->vendor);
	if (resource->version) free(resource->version);

	if (resource->temperature) free(resource->temperature);
#ifdef Intel
	if (resource->sdr) free(resource->sdr);
	if (resource->sensors) free(resource->sensors);
#endif
#ifdef Xilinx
	if (resource->power) free(resource->power);
	if (resource->serial_no) free(resource->serial_no);
	if (resource->topology) free(resource->topology);
#endif

	free(resource);
}

/**
 * Command to execute a compute unit on a resource.
 * @param buffer Refers to a valid compute unit object.
 * @return 0 on success; 1 on failure.
 */
int run_compute_unit(cl_compute_unit compute_unit) {
	int ret = inclEnqueueTask(compute_unit->command_queue, compute_unit->kernel);

	cl_uint num_arguments;
	inclGetKernelInfo(compute_unit->kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &num_arguments, NULL);

	int index;
	for (index = 0; index < num_arguments; index++) {
		if (compute_unit->args[index]) {
			inclSetKernelArg(compute_unit->kernel, index, sizeof(cl_mem), &compute_unit->args[index]);
		}
	}

	return ret;
}

/**
 * Used to set the argument value for a specific argument of a compute unit.
 * @param buffer Refers to a valid compute unit object.
 * @param index The argument index.
 * @param size Specifies the size of the argument value. If the argument is a buffer object, the size is NULL.
 * @param value A pointer to data that should be used for argument specified by index. If the argument is a buffer the value entry will be the appropriate object. The buffer object must be created with the resource associated with the compute unit object.
 * @return 0 on success; 1 on failure.
 */
int set_compute_unit_arg(cl_compute_unit compute_unit, unsigned int index, size_t size, const void *value) {
	if (size) {
		return inclSetKernelArg(compute_unit->kernel, index, size, value);
	} else {
		cl_buffer buffer = (cl_buffer) value;

		if(!compute_unit->args[index]) compute_unit->args[index] = buffer->memory->page_buf;

		return inclSetKernelArg(compute_unit->kernel, index, sizeof(cl_mem), &buffer->mem);
	}
}
