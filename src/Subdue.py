# Subdue.py
#
# Written by Larry Holder (holder@wsu.edu).
#
# Copyright (c) 2017-2021. Washington State University.

import sys
import time
import json
import contextlib
import Parameters
import Graph
import Pattern
import networkx as nx

DEBUGFLAG = False

# ***** todos: read graph file incrementally
def ReadGraph(inputFileName):
    """Read graph from given filename."""
    inputFile = open(inputFileName)
    jsonGraphArray = json.load(inputFile)
    graph = Graph.Graph()
    graph.load_from_json(jsonGraphArray)
    inputFile.close()
    return graph

def createCustomPattern():
    custom_pattern = Graph.Graph()
    
    # Add vertices
    comp_vertex = Graph.Vertex(str(99))
    # comp_vertex.id = str(99)
    comp_vertex.add_attribute("label", "c")
    custom_pattern.vertices[comp_vertex.id] = comp_vertex
    for i in range(0,4):
        wall_vertex = Graph.Vertex(str(99+i+1))
        # wall_vertex.id = str(99+i+1)
        wall_vertex.add_attribute("label", "w")
        custom_pattern.vertices[wall_vertex.id] = wall_vertex
    
    # Add edges
    for i in range(0,4):
        sourceVertex = custom_pattern.vertices[str(99)]
        targetVertex = custom_pattern.vertices[str(99+i+1)]
        cw_edge = Graph.Edge(str(99+i), sourceVertex, targetVertex, False)
        # cw_edge = Graph.Edge()
        # cw_edge.id = str(99+i)
        # cw_edge.source = str(99)
        # cw_edge.target = str(99+i+1)
        cw_edge.add_attribute("label", "e")
        custom_pattern.edges[cw_edge.id] = cw_edge
        sourceVertex.add_edge(cw_edge)
        targetVertex.add_edge(cw_edge)
    
    return custom_pattern


def DiscoverPatterns(parameters, graph):
    """The main discovery loop. Finds and returns best patterns in given graph."""
    patternCount = 0

    custom_pattern = createCustomPattern()

    # get initial one-edge patterns
    parentPatternList = GetInitialPatterns(parameters, graph)
    if DEBUGFLAG:
        print("Initial patterns (" + str(len(parentPatternList)) + "):")
        for pattern in parentPatternList:
            pattern.print_pattern('  ')
    discoveredPatternList = []
    while ((patternCount < parameters.limit) and parentPatternList):
        print(str(int(parameters.limit - patternCount)) + " patterns left", flush=True)
        childPatternList = []
        # extend each pattern in parent list (***** todo: in parallel)
        while (parentPatternList):
            parentPattern = parentPatternList.pop(0)
            for parentInstance in parentPattern.instances:
                parentGraph = Graph.CreateGraphFromInstance(parentInstance)
                if(Graph.GraphMatch(custom_pattern, parentGraph)):
                    print("Custom pattern match found")
            if ((len(parentPattern.instances) > 1) and (patternCount < parameters.limit)):
                patternCount += 1
                extendedPatternList = Pattern.ExtendPattern(parameters, parentPattern)
                while (extendedPatternList):
                    extendedPattern = extendedPatternList.pop(0)
                    if DEBUGFLAG:
                        print("Extended Pattern:")
                        extendedPattern.print_pattern('  ')
                    if (len(extendedPattern.definition.edges) <= parameters.maxSize):
                        # evaluate each extension and add to child list
                        extendedPattern.evaluate(graph)
                        if ((not parameters.prune) or (extendedPattern.value >= parentPattern.value)):
                            Pattern.PatternListInsert(extendedPattern, childPatternList, parameters.beamWidth, parameters.valueBased)
            # add parent pattern to final discovered list
            if (len(parentPattern.definition.edges) >= parameters.minSize):
                Pattern.PatternListInsert(parentPattern, discoveredPatternList, parameters.numBest, False) # valueBased = False
        parentPatternList = childPatternList
        if not parentPatternList:
            print("No more patterns to consider", flush=True)
    # insert any remaining patterns in parent list on to discovered list
    while (parentPatternList):
        parentPattern = parentPatternList.pop(0)
        if (len(parentPattern.definition.edges) >= parameters.minSize):
            Pattern.PatternListInsert(parentPattern, discoveredPatternList, parameters.numBest, False) # valueBased = False
    return discoveredPatternList

def DiscoverStarPatterns(parameters, graph):
    initial_pattern_list = GetInitialStarPatterns(parameters, graph)
    discovered_patterns = []
    for pattern in initial_pattern_list:
        pattern.evaluate(graph)
        Pattern.PatternListInsert(pattern, discovered_patterns, parameters.beamWidth, parameters.valueBased)
    
    return discovered_patterns

def GetInitialPatterns(parameters, graph):
    """Returns list of single-edge, evaluated patterns in given graph with more than one instance."""
    initialPatternList = []
    # Create a graph and an instance for each edge
    edgeGraphInstancePairs = []
    for edge in graph.edges.values():
        graph1 = Graph.CreateGraphFromEdge(edge)
        if parameters.temporal:
            graph1.TemporalOrder()
        instance1 = Pattern.CreateInstanceFromEdge(edge)
        edgeGraphInstancePairs.append((graph1,instance1))
    while edgeGraphInstancePairs:
        edgePair1 = edgeGraphInstancePairs.pop(0)
        graph1 = edgePair1[0]
        instance1 = edgePair1[1]
        pattern = Pattern.Pattern()
        pattern.definition = graph1
        pattern.instances.append(instance1)
        nonmatchingEdgePairs = []
        for edgePair2 in edgeGraphInstancePairs:
            graph2 = edgePair2[0]
            instance2 = edgePair2[1]
            if Graph.GraphMatch(graph1,graph2) and (not Pattern.InstancesOverlap(parameters.overlap, pattern.instances, instance2)):
                pattern.instances.append(instance2)
            else:
                nonmatchingEdgePairs.append(edgePair2)
        if len(pattern.instances) > 1:
            pattern.evaluate(graph)
            initialPatternList.append(pattern)
        edgeGraphInstancePairs = nonmatchingEdgePairs
    return initialPatternList

def GetInitialStarPatterns(parameters, graph):
    initialPatternList = []
    vertexGraphInstancePairs = []
    for vertex in graph.vertices.values():
        new_instance = Pattern.Instance()
        new_instance.vertices.add(vertex)
        for edge in vertex.edges:
            new_instance.edges.add(edge)
            if(edge.source == vertex):
                new_instance.vertices.add(edge.target)
            else:
                new_instance.vertices.add(edge.source)
        graph1 = Graph.CreateGraphFromInstance(new_instance)
        vertexGraphInstancePairs.append((graph1, new_instance))
    while vertexGraphInstancePairs:
        vertexPair1 = vertexGraphInstancePairs.pop(0)
        graph1 = vertexPair1[0]
        instance1 = vertexPair1[1]
        pattern = Pattern.Pattern()
        pattern.definition = graph1
        pattern.instances.append(instance1)
        nonmatchingVertexPairs = []
        for vertexPair2 in vertexGraphInstancePairs:
            graph2 = vertexPair2[0]
            instance2 = vertexPair2[1]
            if Graph.GraphMatch(graph1,graph2) and (not Pattern.InstancesOverlap(parameters.overlap, pattern.instances, instance2)):
                pattern.instances.append(instance2)
            else:
                nonmatchingVertexPairs.append(vertexPair2)
        if len(pattern.instances) > 1:
            pattern.evaluate(graph)
            initialPatternList.append(pattern)
        vertexGraphInstancePairs = nonmatchingVertexPairs
    return initialPatternList


def Subdue(parameters, graph):
    """
    Top-level function for Subdue that discovers best pattern in graph.
    Optionally, Subdue can then compress the graph with the best pattern, and iterate.

    :param graph: instance of Subdue.Graph
    :param parameters: instance of Subdue.Parameters
    :return: patterns for each iteration -- a list of iterations each containing discovered patterns.
    """
    startTime = time.time()
    iteration = 1
    done = False
    patterns = list()
    limit_og = parameters.limit
    while ((iteration <= parameters.iterations) and (not done)):
        iterationStartTime = time.time()
        if (iteration > 1):
            print("----- Iteration " + str(iteration) + " -----\n")
        print("Graph: " + str(len(graph.vertices)) + " vertices, " + str(len(graph.edges)) + " edges")
        # parameters.limit = limit_og / float(iteration*iteration)
        print("limit for this iteration:", parameters.limit)
        patternList = DiscoverPatterns(parameters, graph)
        if (not patternList):
            done = True
            print("No patterns found.\n")
        else:
            patterns.append(patternList)
            print("\nBest " + str(len(patternList)) + " patterns:\n")
            for pattern in patternList:
                pattern.print_pattern('  ')
                print("")
                pattern.analyze(graph)
                print("Class counts:")
                print(pattern.class_counts)
                print("Class probs:")
                print(pattern.class_probs)
            # write machine-readable output, if requested
            if (parameters.writePattern):
                outputFileName = parameters.outputFileName + "-pattern-" + str(iteration) + ".json"
                patternList[0].definition.write_to_file(outputFileName)
            if (parameters.writeInstances):
                outputFileName = parameters.outputFileName + "-instances-" + str(iteration) + ".json"
                patternList[0].write_instances_to_file(outputFileName)
            if ((iteration < parameters.iterations) or (parameters.writeCompressed)):
                graph.Compress(iteration, patternList[0])
            
            patternList[0].updateInstanceVertices()
            doesPatternFit(patternList[0], graph)
            
            if (iteration < parameters.iterations):
                # consider another iteration
                if (len(graph.edges) == 0):
                    done = True
                    print("Ending iterations - graph fully compressed.\n")
            if ((iteration == parameters.iterations) and (parameters.writeCompressed)):
                outputFileName = parameters.outputFileName + "-compressed-" + str(iteration) + ".json"
                graph.write_to_file(outputFileName)
        if (parameters.iterations > 1):
             iterationEndTime = time.time()
             print("Elapsed time for iteration " + str(iteration) + " = " + str(iterationEndTime - iterationStartTime) + " seconds.\n")
        iteration += 1
    endTime = time.time()
    print("SUBDUE done. Elapsed time = " + str(endTime - startTime) + " seconds\n")
    return patterns



def doesPatternFit(pattern, graph):
    # global gMaxMappings
    # print("gMaxMappings", gMaxMappings)
    print("---------")
    # sorted_probs = sorted(pattern.class_probs.items(), key=lambda x:x[1])
    sorted_probs = sorted(pattern.class_probs.items(), key=lambda x:x[1], reverse=True)
    best_class = sorted_probs[0][0]
    print("For pattern")
    pattern.definition.print_graph('  ')
    print("Best class:", best_class)
    start_vertex = None
    # for v in graph.vertices.values():
    #     if v.attributes['label'] == best_class and not v.in_a_pattern:
    #         start_vertex = v
    #         break
    # print("Extending all possible vertices:")
    for v in graph.vertices.values():
        print("Testing")
        v.print_vertex()
        if v.attributes['label'] == best_class and not v.in_a_pattern:
            start_vertex = v
            print("Start vertex: ")
            start_vertex.print_vertex()
            best_subgraph = Graph.Graph()
            best_subgraph.vertices[start_vertex.id] = start_vertex
            subgraph_pattern_maps = []
            for v in pattern.definition.vertices.values():
                if v.attributes['label'] == start_vertex.attributes['label']:
                    map_1 = {start_vertex.id : v.id}
                    subgraph_pattern_maps.append(map_1)
            new_maps_per_struct = []
            extended_subgraphs = Graph.ExtendSubGraphWithPatternCheck(best_subgraph, graph, pattern, subgraph_pattern_maps, new_maps_per_struct)
            print("Extended subgraph:")
            for esg in extended_subgraphs:
                esg.print_graph()
            if len(extended_subgraphs) > 0:
                break
            print("--")
    print("==========================================")
    print("==========================================")
    print("Start vertex: ")
    start_vertex.print_vertex()

    current_structures = []
    # print("Graph vertices:")
    # graph.print_graph()
    for vid in graph.label_maps[best_class]:
        if not vid == start_vertex.id:
            g = Graph.Graph()
            g.vertices[vid] = graph.vertices[vid]
            current_structures.append(g)
    
    best_struct_match_prob = pattern.class_probs[best_class]
    best_subgraph = Graph.Graph()
    best_subgraph.vertices[start_vertex.id] = start_vertex
    subgraph_pattern_maps = []
    for v in pattern.definition.vertices.values():
        if v.attributes['label'] == start_vertex.attributes['label']:
            map_1 = {start_vertex.id : v.id}
            subgraph_pattern_maps.append(map_1)
    print("==========================================")
    _i = 0
    # while best_struct_match_prob < 0.9:
    next_subgraph = best_subgraph
    best_subgraph_pattern_maps = subgraph_pattern_maps
    while _i < 4:
        print("Iteration:", _i)
        _i += 1
        print("Best prob before:", best_struct_match_prob)
        new_maps_per_struct = []
        extended_subgraphs = Graph.ExtendSubGraphWithPatternCheck(next_subgraph, graph, pattern, subgraph_pattern_maps, new_maps_per_struct)
        if len(extended_subgraphs) <= 0:
            print("No more valid extensions. Exiting")
            break

        next_structures = []
        
        print("Extended subgraphs:")
        for esg in extended_subgraphs:
            esg.print_graph()
        
        print("---------------------------")
        
        # for target_subgraph in extended_subgraphs:
        best_prob_this_round = 0.0
        for i in range(0,len(extended_subgraphs)):
            target_subgraph = extended_subgraphs[i]
            print("")
            print("Target subgraph:")
            target_subgraph.print_graph()
            target_subgraph
            target_subgraph_instances = []
            print("Extending current structures")
            for g in current_structures:
                target_subgraph_instances += Graph.ExtendSubGraphTowardsTargetStruct(g, graph, target_subgraph)
                # print("---")
            target_subgraph_instances = Graph.removeDuplicateGraphs(target_subgraph_instances)
            print("Num of extended structs:", len(target_subgraph_instances))
            total_instances = len(target_subgraph_instances)
            pattern_instances = 0
            for inst in target_subgraph_instances:
                # inst.print_graph()
                valid = True
                for v in inst.vertices.values():
                    if not v.in_a_pattern:
                        valid = False
                pattern_instances += int(valid)
                # if valid:
                #     print("In pattern")
                # else:
                #     print("Not in pattern")
            print("Pattern instances:", pattern_instances, "Total instances:", total_instances)
            struct_prob = float(pattern_instances) / float(total_instances)
            print("struct_prob:", struct_prob)
            if struct_prob > best_prob_this_round or struct_prob >= 1.0:
                best_prob_this_round = struct_prob
                next_structures = target_subgraph_instances
                next_subgraph = target_subgraph
                subgraph_pattern_maps = new_maps_per_struct[i]

            if struct_prob > best_struct_match_prob or struct_prob >= 1.0:
                best_struct_match_prob = struct_prob
                best_subgraph = target_subgraph
                best_subgraph_pattern_maps = new_maps_per_struct[i]
        
            print("Best prob after:", best_struct_match_prob)
            print("---------------------------")

        print("Best structure this round (prob", best_prob_this_round, ")")
        next_subgraph.print_graph()
        print("Best structure so far (prob", best_struct_match_prob, ")")
        best_subgraph.print_graph()


        if len(next_structures) <= 0:
            break
        
        
        current_structures = next_structures

        # for i in range(0,len(extended_subgraphs)):
        #     if len(extended_subgraphs) > 0:
        #         best_subgraph = extended_subgraphs[i]
        #         subgraph_pattern_maps = new_maps_per_struct[i]
        #         break
        print("==========================================")
    
    print("==========================================")
    print("Outcome:")
    print("Best subgraph:")
    best_subgraph.print_graph()
    print("With probabilty:", best_struct_match_prob)
    # while best_struct_match_prob < 0.8:
    #     extended_subgraphs = Graph.ExtendSubGraphWithPatternCheck(best_subgraph, graph, pattern, subgraph_pattern_maps)
    #     if len(extended_subgraphs <= 0):
    #         break
    #     next_structures = []
    #     for target_subgraph in extended_subgraphs:
    #         target_subgraph_instances = []
    #         for g in current_structures:
    #             target_subgraph_instances += Graph.ExtendSubGraphTowardsTargetStruct(g, graph, target_subgraph)
    #         target_subgraph_instances = Graph.removeDuplicateGraphs(target_subgraph_instances)
    #         total_instances = len(target_subgraph_instances)
    #         pattern_instances = 0
    #         for inst in target_subgraph_instances:
    #             for v in inst.vertices.values():
    #                 if v.in_a_pattern:
    #                     pattern_instances += 1
    #         struct_prob = float(pattern_instances) / float(total_instances)
    #         if struct_prob > best_struct_match_prob:
    #             best_struct_match_prob = struct_prob
    #             next_structures = target_subgraph_instances
    #             best_subgraph = target_subgraph[0]
    #     if len(next_structures <= 0):
    #         break




def SubdueStar(parameters, graph):
    """
    Top-level function for Subdue that discovers best pattern in graph.
    Optionally, Subdue can then compress the graph with the best pattern, and iterate.

    :param graph: instance of Subdue.Graph
    :param parameters: instance of Subdue.Parameters
    :return: patterns for each iteration -- a list of iterations each containing discovered patterns.
    """
    startTime = time.time()
    iteration = 1
    done = False
    patterns = list()
    while ((iteration <= parameters.iterations) and (not done)):
        iterationStartTime = time.time()
        if (iteration > 1):
            print("----- Iteration " + str(iteration) + " -----\n")
        print("Graph: " + str(len(graph.vertices)) + " vertices, " + str(len(graph.edges)) + " edges")
        patternList = DiscoverStarPatterns(parameters, graph)
        if (not patternList):
            done = True
            print("No patterns found.\n")
        else:
            patterns.append(patternList)
            print("\nBest " + str(len(patternList)) + " patterns:\n")
            for pattern in patternList:
                pattern.print_pattern('  ')
                print("")
            # write machine-readable output, if requested
            if (parameters.writePattern):
                outputFileName = parameters.outputFileName + "-pattern-" + str(iteration) + ".json"
                patternList[0].definition.write_to_file(outputFileName)
            if (parameters.writeInstances):
                outputFileName = parameters.outputFileName + "-instances-" + str(iteration) + ".json"
                patternList[0].write_instances_to_file(outputFileName)
            if ((iteration < parameters.iterations) or (parameters.writeCompressed)):
                graph.Compress(iteration, patternList[0])
            if (iteration < parameters.iterations):
                # consider another iteration
                if (len(graph.edges) == 0):
                    done = True
                    print("Ending iterations - graph fully compressed.\n")
            if ((iteration == parameters.iterations) and (parameters.writeCompressed)):
                outputFileName = parameters.outputFileName + "-compressed-" + str(iteration) + ".json"
                graph.write_to_file(outputFileName)
        if (parameters.iterations > 1):
             iterationEndTime = time.time()
             print("Elapsed time for iteration " + str(iteration) + " = " + str(iterationEndTime - iterationStartTime) + " seconds.\n")
        iteration += 1
    endTime = time.time()
    print("SUBDUE done. Elapsed time = " + str(endTime - startTime) + " seconds\n")
    return patterns

def nx_subdue(
    graph,
    node_attributes=None,
    edge_attributes=None,
    verbose=False,
    **subdue_parameters
):
    """
    :param graph: networkx.Graph
    :param node_attributes: (Default: None)   -- attributes on the nodes to use for pattern matching, use `None` for all
    :param edge_attributes: (Default: None)   -- attributes on the edges to use for pattern matching, use `None` for all
    :param verbose: (Default: False)          -- if True, print progress, as well as report each found pattern

    :param beamWidth: (Default: 4)            -- Number of patterns to retain after each expansion of previous patterns; based on value.
    :param iterations: (Default: 1)           -- Iterations of Subdue's discovery process. If more than 1, Subdue compresses graph with best pattern before next run. If 0, then run until no more compression (i.e., set to |E|).
    :param limit: (Default: 0)                -- Number of patterns considered; default (0) is |E|/2.
    :param maxSize: (Default: 0)              -- Maximum size (#edges) of a pattern; default (0) is |E|/2.
    :param minSize: (Default: 1)              -- Minimum size (#edges) of a pattern; default is 1.
    :param numBest: (Default: 3)              -- Number of best patterns to report at end; default is 3.
    :param overlap: (Defaul: none)            -- Extent that pattern instances can overlap (none, vertex, edge)
    :param prune: (Default: False)            -- Remove any patterns that are worse than their parent.
    :param valueBased: (Default: False)       -- Retain all patterns with the top beam best values.
    :param temporal: (Default: False)         -- Discover static (False) or temporal (True) patterns

    :return: list of patterns, where each pattern is a list of pattern instances, with an instance being a dictionary
    containing 
        `nodes` -- list of IDs, which can be used with `networkx.Graph.subgraph()`
        `edges` -- list of tuples (id_from, id_to), which can be used with `networkx.Graph.edge_subgraph()`
    
    For `iterations`>1 the the list is split by iterations, and some patterns will contain node IDs not present in
    the original graph, e.g. `PATTERN-X-Y`, such node ID refers to a previously compressed pattern, and it can be 
    accessed as `output[X-1][0][Y]`.

    """
    parameters = Parameters.Parameters()
    if len(subdue_parameters) > 0:
        parameters.set_parameters_from_kwargs(**subdue_parameters)
    subdue_graph = Graph.Graph()
    subdue_graph.load_from_networkx(graph, node_attributes, edge_attributes)
    parameters.set_defaults_for_graph(subdue_graph)
    if verbose:
        iterations = Subdue(parameters, subdue_graph)
    else:
        with contextlib.redirect_stdout(None):
            iterations = Subdue(parameters, subdue_graph)
    iterations = unwrap_output(iterations)
    if parameters.iterations == 1:
        if len(iterations) == 0:
            return None
        return iterations[0]
    else:
        return iterations

def unwrap_output(iterations):
    """
    Subroutine of `nx_Subdue` -- unwraps the standard Subdue output into pure python objects compatible with networkx
    """
    out = list()
    for iteration in iterations:
        iter_out = list()
        for pattern in iteration:
            pattern_out = list()
            for instance in pattern.instances:
                pattern_out.append({
                    'nodes': [vertex.id for vertex in instance.vertices],
                    'edges': [(edge.source.id, edge.target.id) for edge in instance.edges]
                })
            iter_out.append(pattern_out)
        out.append(iter_out)
    return out

def main():
    print("SUBDUE v1.4 (python)\n")
    parameters = Parameters.Parameters()
    parameters.set_parameters(sys.argv)
    subdue_graph = ReadGraph(parameters.inputFileName)
    # graph = nx.Graph(nx.nx_pydot.read_dot(parameters.inputFileName))
    # subdue_graph = Graph.Graph()
    # subdue_graph.load_from_networkx(graph)
    # subdue_graph.print_graph()
    #outputFileName = parameters.outputFileName + ".dot"
    #graph.write_to_dot(outputFileName)
    parameters.set_defaults_for_graph(subdue_graph)
    parameters.print()
    Subdue(parameters, subdue_graph)
    # SubdueStar(parameters, graph)

if __name__ == "__main__":
    main()
