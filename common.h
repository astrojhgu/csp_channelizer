/**
 * @file common.h
 * @brief library for common used functions and macros in a CUDA applications
 */

#ifndef _COMMON
#define _COMMON

/**
* @defgroup C typedef for this application
* @{
*/
typedef unsigned char uint_8;
typedef signed char int_8;
/** @} */

/**
 * @brief check if the cuda call correctly worked
 * @param error: return value of a systemcall
 */
#define CHECK_CUDA(error)                                                       \
{                                                                              	\
    if (error != cudaSuccess)                                                  	\
    {                                                                          	\
        fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__);                 	\
        fprintf(stderr, "code: %d, reason: %s\n", error,                       	\
        cudaGetErrorString(error));                                   			\
		exit(-1);						       									\
    }                                                                          	\
    																			\
}


/**
 * @brief check pointer validity
 * @param ptr: generic pointer
 */
#define CHECK_PTR(ptr)                                                          \
{                                                                              	\
    if (ptr == NULL)                                                  			\
    {                                                                          	\
        fprintf(stderr, "Error: %s:%d\n", __FILE__, __LINE__);                 	\
        fprintf(stderr, "Null pointer\n" );				                        \
		exit(-1);						       									\
    }                                                                          	\
    																			\
}


#endif
