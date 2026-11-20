#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>

struct fuse_conn_info_partial {
    unsigned proto_major;
    unsigned proto_minor;
    unsigned max_write;
    unsigned max_read;
    unsigned max_readahead;
    unsigned capable;
    unsigned want;
};

struct fuse_lowlevel_ops_partial {
    void (*init)(void *userdata, struct fuse_conn_info_partial *conn);
};

static void (*real_init)(void *userdata, struct fuse_conn_info_partial *conn) = NULL;

static void wrapped_init(void *userdata, struct fuse_conn_info_partial *conn) {
    if (real_init) {
        real_init(userdata, conn);
    }
    /* Strip FUSE_CAP_CACHE_SYMLINKS (0x800000) and any bit not advertised by the kernel */
    conn->want &= conn->capable;
}

void *fuse_session_new(void *args, struct fuse_lowlevel_ops_partial *op, size_t op_size, void *userdata) {
    static void *(*real_fuse_session_new)(void *, void *, size_t, void *) = NULL;
    if (!real_fuse_session_new) {
        real_fuse_session_new = dlsym(RTLD_NEXT, "fuse_session_new");
    }
    if (op && op->init && op->init != wrapped_init) {
        real_init = op->init;
        op->init = wrapped_init;
    }
    return real_fuse_session_new(args, op, op_size, userdata);
}
