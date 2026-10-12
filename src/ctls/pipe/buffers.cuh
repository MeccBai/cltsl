

template <typename data_t>
class DoubleBuffers
{

    data_t *stage1;
    data_t *stage2;
    bool is_stage1_active;

public:
    __device__ void async_copy(data_t *source, std::size_t count, cudaStream_t stream = nullptr)
    {


    }

    
};
