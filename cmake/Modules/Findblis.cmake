SET(BLIS_INCLUDE_SEARCH_PATHS
	/usr/local/include/blis
)

SET(BLIS_LIB_SEARCH_PATHS
	/usr/local/lib
)

FIND_PATH(BLIS_INCLUDE_DIR NAMES cblas.h blis.h PATHS ${BLIS_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(BLIS_LIB NAMES libblis.so PATHS ${BLIS_LIB_SEARCH_PATHS})

SET(BLIS_FOUND ON)

#    Check include files
IF(NOT BLIS_INCLUDE_DIR)
        SET(BLIS_FOUND OFF)
        MESSAGE(STATUS "Could not find BLIS include. Turning BLIS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT BLIS_LIB)
        SET(BLIS_FOUND OFF)
        MESSAGE(STATUS "Could not find BLIS lib. Turning BLIS_FOUND off")
ENDIF()

IF (BLIS_FOUND)
        IF (NOT BLAS_FIND_QUIETLY)
                MESSAGE(STATUS "Found BLIS libraries: ${BLIS_LIB}")
                MESSAGE(STATUS "Found BLIS include: ${BLIS_INCLUDE_DIR}")
        ENDIF (NOT BLAS_FIND_QUIETLY)
ELSE (BLIS_FOUND)
        IF (BLAS_FIND_REQUIRED)
                MESSAGE(FATAL_ERROR "Could not find BLIS")
        ENDIF (BLAS_FIND_REQUIRED)
ENDIF (BLIS_FOUND)

MARK_AS_ADVANCED(
        BLIS_INCLUDE_DIR
        BLIS_LIB
        blis
)

