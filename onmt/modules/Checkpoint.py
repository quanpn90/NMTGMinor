import torch
from torch.autograd import Variable, Function

def requires_grad(obj):
    
    if not hasattr(obj, 'requires_grad'):
        return False
    
    return obj.requires_grad

def wrap_variable(inputs):
    if torch.is_tensor(inputs):
        return Variable(inputs)
    elif isinstance(inputs, Variable):
        return Variable(inputs.data, requires_grad=True)
    elif isinstance(inputs, tuple):
        return tuple(wrap_variable(v) for v in inputs)
    else:
        raise RuntimeError("Unsupported input type: ", type(inputs).__name__)
        

def unpack_variables(inputs):
    if isinstance(inputs, Variable):
        return inputs.data
    elif torch.is_tensor(inputs):
        return inputs
    elif isinstance(inputs, tuple):
        return tuple(unpack_variables(v) for v in inputs)
    else:
        raise RuntimeError("Unsupported input type: ", type(inputs).__name__)


class CheckpointFunction(Function):

    @staticmethod
    def forward(ctx, run_function, *args):
        ctx.run_function = run_function
        
        # save all of the inputs for the backward pass
        ctx.save_for_backward(*args)
        var_args = wrap_variable(args)
        
        # on the forward pass: don't compute the gradients
        with torch.no_grad():
            outputs = run_function(*var_args)
            #~ ctx.outputs = outputs
            
        out = unpack_variables(outputs)
        
        return out
    
    
    @staticmethod
    def backward(ctx, *grads):
        
        
       	with torch.enable_grad():
        
            real_inputs = ctx.saved_variables
            #~ real_inputs = ctx.inputs
            #~ print(grads[0].size())
            #~ print(len(grads))
            # We need to create new Variables to mark this place in the graph.
            # Reusing real_inputs would be incorrect if a case like this:
            #
            # y = checkpoint(lambda x: x + 1, x)
            # z = checkpoint(lambda x, y: x + y, x, y)
            #
            # This would fail, because when grad((x + y), (x, y)) is called in
            # the second checkpoint, autograd would traverse all paths from (x + y)
            # to the definition of x, which includes the first checkpoint. To
            # prevent this situation, we create views of the inputs, which lets us
            # still get all correctness checks, but uniquely marks the place up to
            # which we want to differentiate, because all views are independent nodes
            # (i.e. there is no path from one to another via .grad_fn chain).
            inputs = [wrap_variable(i) for i in real_inputs]
            outputs = ctx.run_function(*inputs)
                        
            if isinstance(outputs, Variable):
                outputs = (outputs,)

            torch.autograd.backward(outputs, grads)
            
            output = (None,)
            
            for i, input_ in enumerate(inputs):
                
                if requires_grad(input_):
                    
                    if input_.grad is not None:
                        output += (input_.grad, )
                    else:
                        output += (None, )
                
                else:
                    output += (None, )
            
            return output
           


def checkpoint(run_function, *args):
     r"""Checkpoint a model or part of the model
+
+    Checkpoint works by trading compute for memory. It can be applied on any
+    part of the model. In the forward pass, the model is run in volatile
+    manner i.e. the activations are not stored. The forward pass save the
+    inputs tuple and the run_function parameter. In the backwards pass, the
+    saved inputs and run_function is retreived, and the forward pass is done
+    on the model again (non-volatile this time) since we need to get the
+    activations values for calculating the gradient and then the gradients are
+    calculated.
+
+    Args:
+        run_function : describes what to run in the forward pass of the model or
+                       part of the model. It should also know how to handle
+                       the inputs passed as the tuple. For example, in LSTM,
+                       user passes (activation, hidden), run_function should
+                       correctly use first input as activation and second input
+                       as hidden
+        args:         tuple containing inputs to the run_function
+
+    Returns:
+        Output of running the run_function on *args
+    """
     return CheckpointFunction.apply(run_function, *args)
