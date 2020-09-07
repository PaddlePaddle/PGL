""" utils """
import numpy as np
import pgl
import paddle.fluid as fluid

def to_undirected(graph):
    """ to_undirected """
    inv_edges = np.zeros(graph.edges.shape)
    inv_edges[:, 0] = graph.edges[:, 1]
    inv_edges[:, 1] = graph.edges[:, 0]
    
    edges = np.vstack((graph.edges, inv_edges))
    edges = np.unique(edges, axis=0)
#     print(edges.shape)
    g = pgl.graph.Graph(num_nodes=graph.num_nodes, edges=edges)

    for k, v in graph._node_feat.items():
        g._node_feat[k] = v
    return g

def add_self_loop(graph):
    """ add_self_loop """
    self_loop_edges = np.zeros((graph.num_nodes, 2))
    self_loop_edges[:, 0] = self_loop_edges[:, 1]=np.arange(graph.num_nodes)
    edges = np.vstack((graph.edges, self_loop_edges))
    edges = np.unique(edges, axis=0)
#     print(edges.shape)
    g = pgl.graph.Graph(num_nodes=graph.num_nodes, edges=edges)
    
    for k, v in graph._node_feat.items():
        g._node_feat[k] = v
    return g


def linear_warmup_decay(learning_rate, warmup_steps, num_train_steps):
    """ Applies linear warmup of learning rate from 0 and decay to 0."""
    with fluid.default_main_program()._lr_schedule_guard():
        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="scheduled_learning_rate")

        global_step = fluid.layers.learning_rate_scheduler._decay_step_counter()

        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(global_step < warmup_steps):
                warmup_lr = learning_rate * (global_step / warmup_steps)
                fluid.layers.tensor.assign(warmup_lr, lr)
            with switch.default():
                decayed_lr = fluid.layers.learning_rate_scheduler.polynomial_decay(
                    learning_rate=learning_rate,
                    decay_steps=num_train_steps,
                    end_learning_rate=0.0,
                    power=1.0,
                    cycle=False)
                fluid.layers.tensor.assign(decayed_lr, lr)

        return lr, global_step

