
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
    return torch.stack((input0[0] @ input1[0], input0[1] @ input1[0] + input1[1]), dim=0)

if __name__ == '__main__':
    # Example usage
    input_array = torch.tensor([3,1,7,0,4,1,6,3], dtype=torch.float32)
    print("Input Array:", input_array)
    
    # Test with addition
    result = parallel_scan_shitty(input_array.clone())
    print("Parallel Scan Add Result:", result)
    
    expected_result = torch.cumsum(input_array, dim=0)
    print("Expected Result:", expected_result)
    
    assert torch.allclose(result, expected_result), "The parallel scan with addition result does not match the expected result."

    # Test with multiplication
    result = parallel_scan_shitty(input_array.clone(), torch.mul, identity=1)
    print("Parallel Scan Mult Result:", result)

    expected_result = torch.cumprod(input_array, dim=0)
    print("Expected Result:", expected_result)
    assert torch.allclose(result, expected_result), "The parallel scan with multiplication result does not match the expected result."

    # Test with custom binary operation
    A = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
    B = torch.tensor([[5,6],[7,8]], dtype=torch.float32)
    input_array = torch.stack([torch.stack((A,B)), torch.stack((B, A)), torch.stack((A.T, B)), torch.stack((B.T, A))])
    print("Input Array shape:", input_array.shape)
    result = parallel_scan_shitty(input_array.clone(), binary_first_order, identity=torch.tensor([1,0], dtype=torch.float32))
    print("Parallel Scan Custom Result shape:", result.shape)
