import torch
import torch.nn as nn
import types


class TorchGraph(object):
    def __init__(self):
        self._graph = {}
        self.persistence = {}

    def add_tensor_list(self, name, persist=False):
        self._graph[name] = []
        self.persistence[name] = persist

    def append_tensor(self, name, val):
        self._graph[name].append(val)

    def clear_tensor_list(self, name):
        self._graph[name].clear()

    def get_tensor_list(self, name):
        return self._graph[name]

    def clear_all_tensors(self):
        for k in self._graph.keys():
            if not self.persistence[k]:
                self.clear_tensor_list(k)


default_graph = TorchGraph()
default_graph.add_tensor_list('gates_params', True)
default_graph.add_tensor_list('selected_idx')

def apply_func(model, module_type, func, **kwargs):
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            func(m, **kwargs)


def replace_func(model, module_type, func):
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            m.forward = types.MethodType(func, m)


def collect_convbn_gates(m):
    default_graph.append_tensor('gates_params', m.gates)


def init_convbn_gates(m):
    m.gates = nn.Parameter(torch.ones(m.conv.out_channels))


def new_convbn_forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.gates.view(1, -1, 1, 1) * out
    out = self.relu(out)
    return out


def collect_basicblock_gates(m):
    default_graph.append_tensor('gates_params', m.gates)


def init_basicblock_gates(m):
    m.gates = nn.Parameter(torch.ones(m.conv1.out_channels))


def new_basicblock_forward(self, x):
    out = self.bn1(self.conv1(x))
    out = self.gates.view(1, -1, 1, 1) * out
    out = self.bn2(self.conv2(self.relu1(out)))
    out += self.shortcut(x)
    out = self.relu2(out)
    return out


def init_conv_depthwise_gates(m):
    m.gates = nn.Parameter(torch.ones(m.conv2.out_channels))


def collect_conv_depthwise_gates(m):
    default_graph.append_tensor('gates_params', m.gates)


def new_conv_depthwise_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.gates.view(1, -1, 1, 1) * x
    x = self.relu2(x)

    return x


def init_inverted_block_gates(m):
    m.gates = nn.Parameter(torch.ones(m.hid))


def collect_inverted_block_gates(m):
    default_graph.append_tensor('gates_params', m.gates)


def new_inverted_block_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.gates.view(1, -1, 1, 1) * x
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.conv3(x)
    x = self.bn3(x)
    return x


def collect_bottleneck_gates(m):
    default_graph.append_tensor('gates_params', m.gates1)
    default_graph.append_tensor('gates_params', m.gates2)


def init_bottleneck_gates(m):
    m.gates1 = nn.Parameter(torch.ones(m.conv1.out_channels))
    m.gates2 = nn.Parameter(torch.ones(m.conv2.out_channels))


def new_bottleneck_forward(self, x):
    out = self.bn1(self.conv1(x))
    out = self.gates1.view(1, -1, 1, 1) * out
    out = self.bn2(self.conv2(self.relu1(out)))
    out = self.gates2.view(1, -1, 1, 1) * out
    out = self.bn3(self.conv3(self.relu2(out)))
    out += self.shortcut(x)
    out = self.relu3(out)
    return out
