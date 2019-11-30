# কিভাবে 'ওয়েট' অ্যাক্যুরেসি পাল্টাতে পারে?

```text
import numpy as np

# আমাদের প্রেডিকশন করার জন্য ডেটা পয়েন্ট, ছবির সাথে মিলিয়ে দেখুন
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

def relu(input):
    output = max(0, input)
    return output

def predict_with_network(input_data_row, weights):
    node_0_input = (input_data_row * weights['node_0']).sum()
    # print(node_0_input)
    node_0_output = relu(node_0_input)
    # print(node_0_output)

    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)
    
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    return model_output
       

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 0]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1

#print(node_1_input)
#print(node_1_output)
#print(hidden_layer_outputs)
#print(input_to_final_layer)
#print(model_output)


print(model_output_0)
print(model_output_1)
print(error_0)
print(error_1)

```

