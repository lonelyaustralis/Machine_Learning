from Model.Activation_func import Activation_func_gard_check
from Model.Activation_func import GELU, Relu, Sin, tanh, Sigmoid

func = Sigmoid()
# 该函数用于检查激活函数的梯度计算是否正确，第一个参数是激活函数对象，第二参数是进行梯度校验的点位（在激活函数上具体哪个位置上计算梯度）
# 第三个参数用于数值计算的间隔，默认为1e-4
# 返回值第一个是梯度与数值梯度的绝对值之差，第二个是数值梯度除以梯度的值与数字梯度之和的绝对值。当梯度正确时第一个值应该小于1e-9,第二个值应该约等于0.5
# 当出现错误时，建议除了检查激活函数的梯度计算是否正确，还要检查激活函数的前向传播是否正确。
print(Activation_func_gard_check(func, [12]))
