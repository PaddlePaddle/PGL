import pgl
import paddle.fluid as fluid

def DeeperGCN(gw, feature, num_layers, 
        hidden_size, num_tasks, name, dropout_prob):
    """Implementation of DeeperGCN, see the paper
    "DeeperGCN: All You Need to Train Deeper GCNs" in
    https://arxiv.org/pdf/2006.07739.pdf

    Args:
        gw: Graph wrapper object

        feature: A tensor with shape (num_nodes, feature_size)

        num_layers: num of layers in DeeperGCN

        hidden_size: hidden_size in DeeperGCN

        num_tasks: final prediction
        
        name: deeper gcn layer names

        dropout_prob: dropout prob in DeeperGCN

    Return:
        A tensor with shape (num_nodes, hidden_size)
    """
    
    beta = "dynamic"
    feature = fluid.layers.fc(feature,
                     hidden_size,
                     bias_attr=False,
                     param_attr=fluid.ParamAttr(name=name + '_weight'))
    
    output = pgl.layers.gen_conv(gw, feature, name=name+"_gen_conv_0", beta=beta)

    for layer in range(num_layers):
        # LN/BN->ReLU->GraphConv->Res
        old_output = output
        # 1. Layer Norm
        output = fluid.layers.layer_norm(
            output,
            begin_norm_axis=1,
            param_attr=fluid.ParamAttr(
                name="norm_scale_%s_%d" % (name, layer),
                initializer=fluid.initializer.Constant(1.0)),
            bias_attr=fluid.ParamAttr(
                name="norm_bias_%s_%d" % (name, layer),
                initializer=fluid.initializer.Constant(0.0)))

        # 2. ReLU
        output = fluid.layers.relu(output)

        #3. dropout
        output = fluid.layers.dropout(output, 
                dropout_prob=dropout_prob,
                dropout_implementation="upscale_in_train")

        #4 gen_conv
        output = pgl.layers.gen_conv(gw, output, 
                name=name+"_gen_conv_%d"%layer, beta=beta)
        
        #5 res
        output = output + old_output

    # final layer: LN + relu + droput
    output = fluid.layers.layer_norm(
        output,
        begin_norm_axis=1,
        param_attr=fluid.ParamAttr(
            name="norm_scale_%s_%d" % (name, num_layers),
            initializer=fluid.initializer.Constant(1.0)),
        bias_attr=fluid.ParamAttr(
            name="norm_bias_%s_%d" % (name, num_layers),
            initializer=fluid.initializer.Constant(0.0)))
    output = fluid.layers.relu(output)
    output = fluid.layers.dropout(output, 
            dropout_prob=dropout_prob,
            dropout_implementation="upscale_in_train")
    
    # final prediction
    output = fluid.layers.fc(output,
                     num_tasks,
                     bias_attr=False,
                     param_attr=fluid.ParamAttr(name=name + '_final_weight'))

    return output


