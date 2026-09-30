import json
from z3 import *
from antlr4 import *
import networkx as nx
from evotx.utils.solidityParser.antlrGenerated.SolidityLexer import SolidityLexer
from evotx.utils.solidityParser.antlrGenerated.SolidityParser import SolidityParser
from evotx.utils.solidityParser.antlrGenerated.SolidityListener import SolidityListener


class DataFlowListener(SolidityListener):
    def __init__(self):
        super().__init__()
        self.currentFunction = None
        self.dataFlowDict = {}

    def enterFunctionDefinition(self, ctx:SolidityParser.FunctionDefinitionContext):
        if ctx.identifier():
            functionName = ctx.identifier().getText()
            if functionName not in self.dataFlowDict:
                self.currentFunction = functionName          
                self.dataFlowDict[functionName] = {"input_params": {}, "local":{}, "function_call":{}, "graph": nx.DiGraph()}
                for param in ctx.parameterList().parameter():
                    paramName = param.identifier().getText() if param.identifier() else ''
                    paramType = param.typeName().getText()
                    if paramName not in self.dataFlowDict[self.currentFunction]["input_params"]:
                        self.dataFlowDict[self.currentFunction]["input_params"][paramName] = paramType
                    self.dataFlowDict[self.currentFunction]["graph"].add_node(paramName, type=paramType)
        return super().enterFunctionDefinition(ctx)

    def exitFunctionDefinition(self, ctx:SolidityParser.FunctionDefinitionContext):
        self.currentFunction = None
        return super().exitFunctionDefinition(ctx)

    def enterFunctionCall(self, ctx: SolidityParser.FunctionCallContext):
        self._processFunctionCallArguments(ctx.functionCallArguments(), ctx.expression().getText())
        return super().enterFunctionCall(ctx)

    def enterVariableDeclarationStatement(self, ctx:SolidityParser.VariableDeclarationStatementContext):
        if self.currentFunction:
            if ctx.variableDeclarationList():
                for decl in ctx.variableDeclarationList().variableDeclaration():
                    varName = decl.identifier().getText()
                    varType = decl.typeName().getText()
                    if (varName, varType) not in self.dataFlowDict[self.currentFunction]["input_params"]:
                        # self.dataFlowDict[self.currentFunction]["local"].append((varName, varType))
                        if varName not in self.dataFlowDict[self.currentFunction]["local"]:
                            self.dataFlowDict[self.currentFunction]["local"][varName] = varType
                    self.dataFlowDict[self.currentFunction]["graph"].add_node(varName, type=varType)
            elif ctx.variableDeclaration():
                varName = ctx.variableDeclaration().identifier().getText()
                varType = ctx.variableDeclaration().typeName().getText()
                if (varName, varType) not in self.dataFlowDict[self.currentFunction]["input_params"]:
                    # self.dataFlowDict[self.currentFunction]["local"].append((varName, varType))
                    if varName not in self.dataFlowDict[self.currentFunction]["local"]:
                        self.dataFlowDict[self.currentFunction]["local"][varName] = varType
                self.dataFlowDict[self.currentFunction]["graph"].add_node(varName, type=varType)
        return super().enterVariableDeclarationStatement(ctx)

    def enterExpressionStatement(self, ctx:SolidityParser.ExpressionStatementContext):
        # Check if it's an assignment expression
        if self.currentFunction:
            if ctx.expression().getChildCount() == 3:
                self._processExpression(ctx.expression())
            if ctx.expression().functionCallArguments():
                function_name = ctx.expression().expression()[0].getText()
                self._processFunctionCallArguments(ctx.expression().functionCallArguments(), function_name)
            return super().enterExpressionStatement(ctx)
    
    def enterRequireStatement(self, ctx: SolidityParser.RequireStatementContext):
        if self.currentFunction:
            if ctx.expression()[0].getChildCount() == 3:
                self._processExpression(ctx.expression()[0], _label="require")
        return super().enterRequireStatement(ctx)
    
    def enterAssemblyIf(self, ctx: SolidityParser.AssemblyIfContext):
        return super().enterAssemblyIf(ctx)
    
    def _processFunctionCallArguments(self, ctx: SolidityParser.FunctionCallArgumentsContext, function_name: str):
        if function_name.startswith('emit'):
            return
        if self.currentFunction:
            if function_name not in self.dataFlowDict[self.currentFunction]["function_call"]:
                self.dataFlowDict[self.currentFunction]["function_call"][function_name] = []
            if ctx.expressionList():
                temp = []
                for arg in ctx.expressionList().expression():
                    temp.append(arg.getText())
                if len(temp) > 0:
                    self.dataFlowDict[self.currentFunction]["function_call"][function_name].append(temp)

    
    def _processExpression(self, ctx, _label=''):
        # Placeholder for a method to process expressions and extract variable names
        if self.currentFunction:
            self.dataFlowDict[self.currentFunction]["graph"].add_edge(ctx.expression(0).getText(), ctx.expression(1).getText(), operator=ctx.getChild(1).getText(), label=_label)

    def getGraph(self, functionName):
        data_info = self.dataFlowDict[functionName]
        complete_graph = data_info['graph']
        for call in data_info['function_call']:
            for args in data_info['function_call'][call]:
                mapping = {}
                if call in self.dataFlowDict:
                    call_graph = self.getGraph(call)
                    input_params = list(self.dataFlowDict[call]['input_params'].keys())
                    for i in range(len(args)):
                        mapping[f"{input_params[i]}"] = args[i]
                    function_graph = nx.relabel_nodes(call_graph, mapping, copy=True)
                    complete_graph = nx.compose(complete_graph, function_graph)
        return complete_graph    
    
    def getFunctionCall(self, functionName):
        return self.dataFlowDict[functionName]["function_call"]
    
    def getInputParams(self, functionName):
        return self.dataFlowDict[functionName]["input_params"]
                
                    
def get_edges_by_label(graph, label):
    return [(u, v,d) for u, v, d in graph.edges(data=True) if d.get('label') == label]

# Add more overrides as necessary to capture the full data flow
def parse_solidity_code(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
        lexer = SolidityLexer(InputStream(code))
        stream = CommonTokenStream(lexer)
        parser = SolidityParser(stream)
        tree = parser.sourceUnit()  # Adjust according to your entry point
        listener = DataFlowListener()
        walker = ParseTreeWalker()
        walker.walk(listener, tree)
        return listener
    

def get_path_from_graph(graph, start_node, end_node):
    if nx.has_path(graph, start_node, end_node):
        path = nx.shortest_path(graph, source=start_node, target=end_node)
        edges_with_info = []
        for i in range(len(path)-1):
            edge_info = graph.get_edge_data(path[i], path[i+1])
            edges_with_info.append((path[i], path[i+1], edge_info))        
        return edges_with_info
    return None

def retrieve_related_edge(graph, edge, input_params):
    output = []
    for end_node in edge[:2]:
        if end_node not in input_params:
            for start_node in input_params:
                path = get_path_from_graph(graph, start_node, end_node)
                if path is not None:
                    if len(path) == 1:
                        if path[0] != edge:
                            output.extend(path)
                    else:
                        output.extend(path)
    return output


def extract_contraints(input_path, key_func=None):
    listener = parse_solidity_code(input_path)
    if key_func is None:
        key_func = list(listener.dataFlowDict.keys())[0]
    graph = listener.getGraph(key_func)
    input_params = listener.getInputParams(key_func)
    constraints = get_edges_by_label(graph, 'require')
    constraint_output = []
    for edge in constraints:
        constraint_output.append(edge)
        related_edge = retrieve_related_edge(graph, edge, input_params)
        if len(related_edge) > 0:
            constraint_output.extend(related_edge)
    return {
        'function_name': key_func,
        'input_params': input_params,
        'constraints': constraint_output
    }
            
def extract_contraints_path(input_path):
    key_func, input_params, constraint_output = extract_contraints(input_path)
    output = {
        'function_name': key_func,
        'input_params': input_params,
        'constraints': constraint_output
    }
    return output

def extract_contraints_path_to_json(input_root, output_path):
    output = {}
    for file in os.listdir(input_root):
        if file.endswith('.sol'):
            key_func, input_params, constraint_output = extract_contraints(os.path.join(input_root, file))
            output[file.replace('_exp.sol.sol','')] = {
                'function_name': key_func,
                'input_params': input_params,
                'constraints': constraint_output
            }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    # graph = parse_solidity_code('./data/ac_updated_code/dodo_flashloan_exp.sol.sol')
    extract_contraints_path_to_json('./data/ac_updated_code', './data/ac_updated_code_constraints.json')
