#import onnx
#
#model = onnx.load("your_model.onnx")
#total_params = 0
#
#for tensor in model.graph.initializer:
#    param_count = 1
#    for dim in tensor.dims:
#        param_count *= dim
#    total_params += param_count
#
#print(f"Total parameters: {total_params}")

import onnx_tool
#from onnx_tool import calculate_flops
out = onnx_tool.model_profile("/home/max/ieos/small_obj/vid_pred/onnx_models/lodki_v2_640x640.onnx")
#print(f"FLOPs: {flops}")
#print(f"Parameters: {params}")

print(f'out: \n{out}')