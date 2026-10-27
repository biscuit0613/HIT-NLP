"""在 GPU 节点上验证 paddle GPU 可用性。用法: srun ... python check_paddle.py"""
import paddle

print("paddle", paddle.__version__)
print("compiled_with_cuda", paddle.is_compiled_with_cuda())
print("gpu count", paddle.device.cuda.device_count())
paddle.set_device("gpu:0")
x = paddle.randn([2048, 2048])
y = paddle.matmul(x, x)
print("matmul ok", y.shape, y.dtype, float(y.abs().mean()))
print("gpu name", paddle.device.cuda.get_device_name(0))
