# SPDX-License-Identifier: Apache-2.0
"""Partition a retained HIP graph at an empty dependency marker.

Only graph topology changes. Kernels, their arguments, allocations and BLAS
algorithm choices are inherited from the original graph, which stays alive.
"""
import ctypes as C


class HipGraphs:
    def __init__(self):
        self.lib = C.CDLL("libamdhip64.so")
        pointer = C.c_void_p
        array = C.POINTER(pointer)
        size = C.POINTER(C.c_size_t)
        signatures = {
            "hipGraphGetNodes": [pointer, array, size],
            "hipGraphGetEdges": [pointer, array, array, size],
            "hipGraphNodeGetType": [pointer, C.POINTER(C.c_int)],
            "hipGraphClone": [array, pointer],
            "hipGraphNodeFindInClone": [array, pointer, pointer],
            "hipGraphDestroyNode": [pointer],
            "hipGraphDestroy": [pointer],
            "hipGraphExecDestroy": [pointer],
            "hipGraphInstantiate": [array, pointer, array, pointer, C.c_size_t],
            "hipGraphLaunch": [pointer, pointer],
            "hipStreamGetCaptureInfo_v2": [
                pointer, C.POINTER(C.c_int), C.POINTER(C.c_ulonglong),
                array, C.POINTER(array), size,
            ],
            "hipGraphAddEmptyNode": [array, pointer, array, C.c_size_t],
            "hipStreamUpdateCaptureDependencies": [pointer, array, C.c_size_t, C.c_uint],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        self.lib.hipGetErrorString.argtypes = [C.c_int]
        self.lib.hipGetErrorString.restype = C.c_char_p

    def call(self, name, *args):
        error = getattr(self.lib, name)(*args)
        if error:
            reason = self.lib.hipGetErrorString(error).decode()
            raise RuntimeError(f"{name}: {error} ({reason})")

    def marker(self, stream):
        status, capture_id = C.c_int(), C.c_ulonglong()
        graph, deps, count = C.c_void_p(), C.POINTER(C.c_void_p)(), C.c_size_t()
        self.call("hipStreamGetCaptureInfo_v2", stream, C.byref(status),
                  C.byref(capture_id), C.byref(graph), C.byref(deps), C.byref(count))
        if status.value != 1 or not graph.value:
            raise RuntimeError("Step5 boundary marker requires an active stream capture")
        dependencies = (C.c_void_p * count.value)(*[deps[i] for i in range(count.value)])
        marker = C.c_void_p()
        self.call("hipGraphAddEmptyNode", C.byref(marker), graph, dependencies, count)
        dependency = (C.c_void_p * 1)(marker.value)
        # Set (replace), rather than add, the stream's current dependency set.
        self.call("hipStreamUpdateCaptureDependencies", stream, dependency, 1, 0)
        return graph.value, marker.value

    def topology(self, graph):
        count = C.c_size_t()
        self.call("hipGraphGetNodes", graph, None, C.byref(count))
        nodes = (C.c_void_p * count.value)()
        self.call("hipGraphGetNodes", graph, nodes, C.byref(count))
        types = {}
        for node in nodes:
            kind = C.c_int()
            self.call("hipGraphNodeGetType", node, C.byref(kind))
            # DTK's stream-captured Torch/HSA kernels have vendor type200.
            # They remain opaque: clone their native nodes and never rewrite
            # kernel parameters. The multistream replay test audits this path.
            # Host callbacks, external semaphores, graph allocations and
            # conditional nodes require a separate ownership/dependency audit.
            # Audited immediately before partitioning, after collecting all nodes.
            types[node] = kind.value
        edges = C.c_size_t()
        self.call("hipGraphGetEdges", graph, None, None, C.byref(edges))
        source, target = (C.c_void_p * edges.value)(), (C.c_void_p * edges.value)()
        self.call("hipGraphGetEdges", graph, source, target, C.byref(edges))
        return types, list(zip(source, target))

    def clone_subset(self, graph, nodes, keep):
        clone = C.c_void_p()
        self.call("hipGraphClone", C.byref(clone), graph)
        try:
            # Find every clone node before destroying anything: cloning lookup
            # must not depend on an already-mutated topology.
            mapped = {}
            for node in nodes:
                result = C.c_void_p()
                self.call("hipGraphNodeFindInClone", C.byref(result), node, clone)
                mapped[node] = result.value
            for node, copied in mapped.items():
                if node not in keep:
                    self.call("hipGraphDestroyNode", copied)
            executable = C.c_void_p()
            self.call("hipGraphInstantiate", C.byref(executable), clone, None, None, 0)
            return NativeGraph(self, clone.value, executable.value)
        except BaseException:
            self.call("hipGraphDestroy", clone)
            raise


class NativeGraph:
    def __init__(self, api, graph, executable):
        self.api, self.graph, self.executable = api, graph, executable

    def replay(self, stream):
        self.api.call("hipGraphLaunch", self.executable, stream)

    def close(self):
        if self.executable:
            self.api.call("hipGraphExecDestroy", self.executable)
            self.executable = None
        if self.graph:
            self.api.call("hipGraphDestroy", self.graph)
            self.graph = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def partition(api, graph, marker):
    types, edges = api.topology(graph)
    if any(t not in (0, 1, 2, 4, 5, 6, 7, 200) for t in types.values()):
        from collections import Counter
        raise RuntimeError(f"Unsupported Step5 graph topology: {dict(Counter(types.values()))}")
    if types.get(marker) != 5:
        raise RuntimeError("Step5 graph boundary is not the expected empty node")
    parents = {node: [] for node in types}
    for source, target in edges:
        parents[target].append(source)
    ancestors, todo = set(), list(parents[marker])
    while todo:
        node = todo.pop()
        if node not in ancestors:
            ancestors.add(node)
            todo.extend(parents[node])
    suffix = set(types) - ancestors - {marker}
    if not ancestors or not suffix:
        raise RuntimeError("Step5 graph boundary produced an empty partition")
    if any(source in suffix and target in ancestors for source, target in edges):
        raise RuntimeError("Step5 prefix is not closed under graph dependencies")
    prefix_graph = api.clone_subset(graph, types, ancestors)
    try:
        suffix_graph = api.clone_subset(graph, types, suffix)
    except BaseException:
        prefix_graph.close()
        raise
    return prefix_graph, suffix_graph, dict(
        original_nodes=len(types), prefix_nodes=len(ancestors),
        suffix_nodes=len(suffix), marker_nodes=1,
    )
