
import torch
import math

def up_sweep(input_array, binary_operation=torch.add):
    """
    Perform the up-sweep (reduce) phase of the parallel scan algorithm.
    This function computes the prefix sum of the input array in parallel.
    """
    for i in range(int(math.log2(len(input_array)))):
        for j in range(0, len(input_array), 2**(i+1)):
            input_array[j + 2**(i+1) - 1] = binary_operation(input_array[j + 2**(i+1) - 1], input_array[j + 2**i - 1])
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
    # Step 1: Up-sweep phase
    up_sweep_result = up_sweep(input_array, binary_operation)
    
    # Step 2: Down-sweep phase
    final_result = down_sweep(up_sweep_result, binary_operation, identity=identity)
    
    return final_result

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
    @staticmethod
    def forward(ctx, A_bar, B_bar, u):
        """
        A_bar : (L, N)
        B_bar : (L, N)
        """
        
        ctx.save_for_backward(A_bar, u)
        B_bar = B_bar * u
        input_array = torch.stack([torch.stack((A_bar1, B_bar1)) for A_bar1, B_bar1 in zip(A_bar, B_bar)])

        out = parallel_scan_shitty(input_array, binary_first_order, identity=torch.stack([torch.ones(A_bar.shape[-1]), torch.zeros(B_bar.shape[-1])]))
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        #in progress :)
        A_bar = ctx.saved_tensors
        dl_dx = grad_output[:, 0]
        breakpoint()
        
        return 1, 2, 3

if __name__ == '__main__':
    A_bar = torch.randn(4, 3, requires_grad=True)
    B_bar = torch.randn(4, 3, requires_grad=True)
    u = torch.randn(4, 3, requires_grad=True)

    result = parallel_scan_naive.apply(A_bar, B_bar, u)

    # Let’s reduce it to a scalar for backward()
    loss = result.sum()
    loss.backward()

    print("Gradient wrt A_bar:")
    print(A_bar.grad)
    
    print("Gradient wrt B_bar:")
    print(B_bar.grad)
