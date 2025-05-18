
import torch
import math

def up_sweep(input_array, binary_operation=torch.add):
    """
    Perform the up-sweep (reduce) phase of the parallel scan algorithm.
    This function computes the prefix sum of the input array in parallel.
    """
    for i in range(int(math.log2(len(input_array)))):
        for j in range(0, len(input_array), 2**(i+1)):
            input_array[j + 2**(i+1) - 1] = binary_operation(input_array[j + 2**i - 1], input_array[j + 2**(i+1) - 1])
    return input_array

def down_sweep(input_array, binary_operation=torch.add, identity=0):
    """
    Perform the down-sweep phase of the parallel scan algorithm.
    This function computes the final prefix sum using the results from the up-sweep phase.
    """    
    sum = input_array[-1].clone()  # Store the last element for the down-sweep phase
    input_array[-1] = identity # Set the last element to zero for the down-sweep phase
    for i in range(int(math.log2(len(input_array)))-1, -1,-1):
        for j in range(0, len(input_array), 2**(i+1)):
            temp = input_array[j + 2**i - 1].clone()
            input_array[j + 2**i - 1] = input_array[j + 2**(i+1) - 1] #set left child
            input_array[j + 2**(i+1) - 1] = binary_operation(input_array[j + 2**(i+1) - 1],  temp) #set right child
    input_array[:-1] = input_array[1:].clone()  # Restore the last element to its original value
    input_array[-1] = sum  # Set the first element to the total sum
    return input_array

def parallel_scan_shitty(input_array, binary_operation=torch.add, identity=0):
    """
    Perform the parallel scan (prefix sum) operation on the input array.
    This function combines the up-sweep and down-sweep phases to compute the prefix sum.
    """

    #This code is ugly but takes care of when input_array_len // 2**j != 0
    padding_len = is_divisible_by_2(input_array.shape[0])[1]
    original_len = input_array.shape[0]
    if padding_len != 0:
        padding_len = 2**padding_len - input_array.shape[0]
        
        padding_shape = (padding_len, 1, *input_array.shape[2:])
        padding_zeros = torch.zeros(padding_shape, dtype=input_array.dtype, device=input_array.device)
        padding_ones = torch.ones(padding_shape, dtype=input_array.dtype, device=input_array.device)
        input_array = torch.cat([input_array, torch.cat([padding_ones, padding_zeros], dim = 1)])

    # Step 1: Up-sweep phase
    up_sweep_result = up_sweep(input_array, binary_operation)
    
    # Step 2: Down-sweep phase
    final_result = down_sweep(up_sweep_result, binary_operation, identity=identity)
    return final_result[:original_len]

def is_divisible_by_2(x: int) -> bool:
    j = 1
    while 2**j < x:
        j += 1
    return (x & ((1 << j) - 1)) == 0, j

def binary_first_order(input0, input1):
    """
    A binary first-order function that combines two inputs.
    """
    return torch.stack((input0[0] * input1[0], input0[1] * input1[0] + input1[1]), dim=0)

def test_parallel_scan():
    A = torch.tensor([1, 2], dtype=torch.float32)
    B = torch.tensor([5, 6], dtype=torch.float32)
    input_array = torch.stack([torch.stack((A,B)), torch.stack((B, A)), torch.stack((A.T, B)), torch.stack((B.T, A))])
    print("Input Array shape:", input_array.shape)
    result = parallel_scan_shitty(input_array.clone(), binary_first_order, identity=torch.tensor([1,0], dtype=torch.float32))
    print("Parallel Scan Custom Result shape:", result.shape)

class parallel_scan_naive(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs, output):
       A_bar, B_bar, u, C = inputs
       ctx.save_for_backward(A_bar, B_bar, C, u)
       ctx.mark_non_differentiable(u)

    @staticmethod
    def forward(A_bar, B_bar, u, C):
        """
        A_bar : (L, N)
        B_bar : (L, N)
        C : (L, N)
        u : (L)
        """
        assert len(A_bar) == len(B_bar) - 1, "first element of A_bar should be ID."
        A_bar = torch.cat([torch.ones((1, *A_bar.shape[1:])), A_bar], dim=0)

        B_bar = B_bar * u.unsqueeze(-1)
        input_array = torch.stack([torch.stack((A_bar1, B_bar1)) for A_bar1, B_bar1 in zip(A_bar, B_bar)])
        x_states = parallel_scan_shitty(input_array, binary_first_order, identity=torch.stack([torch.ones(A_bar.shape[-1]), torch.zeros(B_bar.shape[-1])]))
        y = [torch.sum(c * xk) for c, xk in zip(C, x_states[:,1])]
        return torch.stack(y)
    
    @staticmethod
    def backward(ctx, grad_output):
        A_bar, B_bar, C, u = ctx.saved_tensors
        A_flip = torch.cat([torch.ones((1, *A_bar.shape[1:])), torch.flip(A_bar, dims=[0])], dim=0) 
        A_bar = torch.cat([torch.ones((1, *A_bar.shape[1:])), A_bar], dim=0)
        dl_dx = grad_output.unsqueeze(-1) * C

        recompute_array = torch.stack([torch.stack((A_bar1, B_bar1)) for A_bar1, B_bar1 in zip(A_bar, B_bar * u.unsqueeze(-1))])
        x_states = parallel_scan_shitty(recompute_array, binary_first_order, identity=torch.stack([torch.ones(A_bar.shape[-1]), torch.zeros(B_bar.shape[-1])]))[:, 1]

        input_array = torch.stack([torch.stack((A_bar, dl_dx)) for A_bar, dl_dx in zip((A_flip), torch.flip(dl_dx, dims=[0]))])
        grad_x = parallel_scan_shitty(input_array, binary_first_order, identity=torch.stack([torch.ones(A_bar.shape[-1]), torch.zeros(dl_dx.shape[-1])]))[:, 1]
        grad_x = torch.flip(grad_x, dims=[0])
        shifted_x_states = torch.cat([torch.zeros(x_states.shape[1]).to(x_states.device).unsqueeze(0), x_states[:-1]], dim=0)
        grad_A_bar = shifted_x_states * grad_x
        
        grad_B_bar = u.unsqueeze(-1) * grad_x
        grad_u = None
        grad_C = grad_output.unsqueeze(-1) * x_states
        
        return grad_A_bar[1:], grad_B_bar, grad_u, grad_C

if __name__ == '__main__':
    A_bar = torch.tensor([[3,1,2], [5,1,1]], dtype=torch.float32, requires_grad=True)
    B_bar = torch.tensor([[6,1,2], [9,8,3], [3,4,6]], dtype=torch.float32, requires_grad=True)
    C = torch.tensor([[1,2,3], [4,5,7], [1,2,6]], dtype=torch.float32, requires_grad=True)
    u = torch.tensor([5,8,3], dtype=torch.float32, requires_grad=True)

    result = parallel_scan_naive.apply(A_bar, B_bar, u, C)

    # Let’s reduce it to a scalar for backward()
    loss = result.sum()
    loss.backward()

    print("Gradient wrt A_bar:")
    print(A_bar.grad)
    
    print("Gradient wrt B_bar:")
    print(B_bar.grad)
