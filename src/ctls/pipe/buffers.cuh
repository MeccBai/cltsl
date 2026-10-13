

template <typename data_t, unsigned int size>
class DoubleBuffers
{
    data_t *stage1;
    data_t *stage2;
    bool is_stage1_active;

public:
    __device__ DoubleBuffers(data_t *stage1, data_t *stage2) : stage1(stage1), stage2(stage2), is_stage1_active(true)
    {
    }

    __device__ void async_copy(data_t *source, std::size_t count)
    {
    }

    __device__ void wait_for_ready()
    {
    }
};
