#ifndef RCCL1_COMPAT_H
#define RCCL1_COMPAT_H

#include <rccl.h>

#ifndef RCCL_MAJOR // RCCL 1.x
#define RCCL_MAJOR 1
#define RCCL_MINOR 0

#define rcclNumOps rccl_NUM_OPS
#define rcclNumTypes rccl_NUM_TYPES

 static rcclResult_t rcclGroupStart() { return rcclSuccess; }
static rcclResult_t rcclGroupEnd() { return rcclSuccess; }

#define CHECKCOUNT(count) if (count > INT_MAX) return rcclInvalidArgument;

 /*
static rcclResult_t rcclReduce(const void* sendbuff, void* recvbuff, size_t count, rcclDataType_t datatype,
    rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
  CHECKCOUNT(count);
  return rcclReduce(sendbuff, recvbuff, (int)count, datatype, op, root, comm, stream);
}
static rcclResult_t rcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    rcclDataType_t datatype, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
  CHECKCOUNT(count);
  return rcclAllReduce(sendbuff, recvbuff, (int)count, datatype, op, comm, stream);
}
static rcclResult_t rcclBcast(void* buff, size_t count, rcclDataType_t datatype, int root,
    rcclComm_t comm, hipStream_t stream) {
  CHECKCOUNT(count);
  return rcclBcast(buff, (int)count, datatype, root, comm, stream);
}
static rcclResult_t rcclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, rcclDataType_t datatype, rcclRedOp_t op, rcclComm_t comm,
    hipStream_t stream) {
  CHECKCOUNT(recvcount);
  return rcclReduceScatter(sendbuff, recvbuff, (int)recvcount, datatype, op, comm, stream);
}
*/
static rcclResult_t rcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    rcclDataType_t datatype, rcclComm_t comm, hipStream_t stream) {
  CHECKCOUNT(sendcount);
  return rcclAllGather(sendbuff, (int)sendcount, datatype, recvbuff, comm, stream);
}
#endif

#endif
