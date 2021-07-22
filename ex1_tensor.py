import torch

# 1. Scalar
scalar1 = torch.tensor([1.])
scalar2 = torch.tensor([.2])

print('=====================================')

# 2. Vector
vector1 = torch.tensor([1., 2., 3.])
vector2 = torch.tensor([4., 5., 6.])

add_vector = vector1 + vector2
add_vector_m = torch.add(vector1, vector2)
print(f'Vector 덧셈 연산 without module : {add_vector}')
print(f'Vector 덧셈 연산 with module : {add_vector_m}')

print('=====================================')

# 3. Matrix
matrix1 = torch.tensor([[1., 2.], [3., 4.]])
matrix2 = torch.tensor([[5., 6.], [7., 8.]])

mul_matrix = matrix1 * matrix2
mul_matrix_m = torch.mul(matrix1, matrix2)
mul_matrix_mm = torch.matmul(matrix1, matrix2)
print(f'요소별 곱셈 연산 without module : {mul_matrix}')
print(f'요소별 곱셈 연산 with module : {mul_matrix_m}')
print(f'Matrix 곱셈 연산 module : {mul_matrix_mm}')

print('=====================================')

# 4. Tensor
tensor1 = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.,]]])
tensor2 = torch.tensor([[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.,]]])

mul_tensor = tensor1 * tensor2
mul_tensor_m = torch.mul(tensor1, tensor2)
mul_tensor_mm = torch.matmul(tensor1, tensor2)
print(f'요소별 곱셈 연산 without module : {mul_tensor}')
print(f'요소별 곱셈 연산 with module : {mul_tensor_m}')
print(f'Tensor 곱셈 연산 module : {mul_tensor_mm}')
# [[(1*9)+(2*11)], [(1*10)+(2*12)], [(5*13)+(6*15)], [(5*14)+(6*16)],
# [(3*9)+(4*11)], [(3*19)+(4*12)], [(7*13)+(8*15)], [(7*14)+(8*16)]
