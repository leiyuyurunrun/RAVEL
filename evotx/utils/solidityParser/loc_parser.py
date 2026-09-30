import os
import json
import antlr4
from tqdm import tqdm
from evotx.utils.solidityParser.antlrGenerated.SolidityLexer import SolidityLexer
from evotx.utils.solidityParser.antlrGenerated.SolidityParser import SolidityParser
from evotx.utils.solidityParser.antlrGenerated.SolidityListener import SolidityListener
from eth_utils import keccak, to_hex
 


class CustomizedSolidityListener(SolidityListener):
    def __init__(self):
        self.current_contract = None
        self.isMainContract = False
        self.last_line_of_import = -1
        self.first_line_of_state_variables = -1
        self.last_line_of_state_variables = -1
        self.first_line_of_function = -1
        self.contracts = {}
        self.first_contract = True

    def get_position(self, ctx):
        return ctx.start.line, ctx.stop.line if ctx.stop else ctx.start.line
    
    def enterImportDirective(self, ctx: SolidityParser.ImportDirectiveContext):
        if len(self.contracts) == 1:
            if ctx.stop:
                if ctx.stop.line > self.last_line_of_import:
                    self.last_line_of_import = ctx.stop.line
        return super().enterImportDirective(ctx)

    def enterContractDefinition(self, ctx:SolidityParser.ContractDefinitionContext):
        if self.first_contract:
            self.first_contract = False
            self.last_line_of_import = ctx.start.line - 1
        contract_name = ctx.identifier().getText()
        contract_type = 'contract'
        if ctx.InterfaceKeyword():
            contract_type = 'interface'
        elif ctx.LibraryKeyword():
            contract_type = 'library'
        elif ctx.AbstractKeyword():
            contract_type = 'abstract'
        self.current_contract = contract_name
        self.contracts[contract_name] = {
            "start_line": ctx.start.line,
            "end_line": None,
            'type': contract_type,
            "functions": {},
            "events": [],
            "state_variables": []
        }
        self.first_line_of_state_variables = -1
        self.last_line_of_state_variables = -1
        self.first_line_of_function = -1

    def exitContractDefinition(self, ctx:SolidityParser.ContractDefinitionContext):
        self.contracts[self.current_contract]["end_line"] = ctx.stop.line
        if self.last_line_of_state_variables < self.first_line_of_function:
            self.contracts[self.current_contract]["state_variables_first"] = True
        else:
            self.contracts[self.current_contract]["state_variables_first"] = False
        self.contracts[self.current_contract]["state_variables_loc"] = {
            'start_line': self.first_line_of_state_variables,
            'end_line': self.last_line_of_state_variables
        }
        # if self.isMainContract:
        #     self.contracts[self.current_contract]["isMainContract"] = True
        #     self.contracts['ContractTest'] = self.contracts[self.current_contract]
        #     if self.current_contract != 'ContractTest':
        #         del self.contracts[self.current_contract]
        # self.isMainContract = False
        self.current_contract = None

    def enterFunctionDefinition(self, ctx:SolidityParser.FunctionDefinitionContext):
        if self.first_line_of_function == -1:
            self.first_line_of_function = ctx.start.line
        if ctx.identifier():
            function_name = ctx.identifier().getText()
        else:
            function_name = ''
        # if function_name.startswith('test'):
        #     self.isMainContract = True
        start_line, end_line = self.get_position(ctx)
        if self.current_contract:
            self.contracts[self.current_contract]["functions"][function_name] = {
                "start_line": start_line,
                "end_line": end_line
            }
            if ctx.block():
                child_count = ctx.block().getChildCount()
                if child_count <= 2:
                    return
                block_start_line = ctx.block().getChild(1).start.line
                block_end_line = ctx.block().getChild(child_count-2).stop.line if ctx.block().getChild(child_count-2).stop else ctx.block().getChild(child_count-2).start.line
                self.contracts[self.current_contract]["functions"][function_name]["block"] = {
                    "start_line": block_start_line,
                    "end_line": block_end_line
                }

    def enterEventDefinition(self, ctx:SolidityParser.EventDefinitionContext):
        self.contracts[self.current_contract]["events"].append(ctx.getText())
        
    def _compute_function_signature(function_name):
        signature = "transfer(address,uint256)"
        hash_bytes = keccak(text=signature)
        return to_hex(hash_bytes[:4])  

    # def enterStateVariableDeclaration(self, ctx:SolidityParser.StateVariableDeclarationContext):
    #     # This assumes the state variable is declared with an identifier.
    #     # You may need to adjust this depending on how your state variables are declared.
    #     if ctx.identifier():
    #         variable_name = ctx.identifier().getText()
    #         start_line, end_line = self.get_position(ctx)
    #         if self.first_line_of_state_variables == -1:
    #             self.first_line_of_state_variables = start_line
    #         if end_line > self.last_line_of_state_variables:
    #             self.last_line_of_state_variables = end_line
    #         variable_info = {
    #             "name": variable_name,
    #             "type": ctx.typeName().getText(),
    #             "start_line": start_line,
    #             "end_line": end_line
    #         }

    #         if self.current_contract:
    #             self.contracts[self.current_contract]["state_variables"].append(variable_info)
    #         else:
    #             self.contracts['global_state_variables'].append(variable_info)
    
    def get_contracts_info(self):
        # self.contracts['last_line_of_import'] = self.last_line_of_import
        return self.contracts



def antlr_listener(f):
    with open(f, 'r',encoding='utf-8') as file:
        code = file.read()

    input_stream = antlr4.InputStream(code)
    lexer = SolidityLexer(input_stream)
    lexer.removeErrorListeners()
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = SolidityParser(token_stream)
    parser.removeErrorListeners()
    tree = parser.sourceUnit()

    listener = CustomizedSolidityListener()
    walker = antlr4.ParseTreeWalker()
    walker.walk(listener, tree)

    return listener

def parse(f):
    listener = antlr_listener(f)
    return listener.get_contracts_info()


def parse_batch(input_dir, output_path):
    output = {}
    for f in tqdm(os.listdir(input_dir)):
        if (f.endswith('.sol')) and (f != 'interface.sol'):
            output[f.replace('.sol','')] = parse(os.path.join(input_dir, f))
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

def get_loc_info(code):
    input_stream = antlr4.InputStream(code)
    lexer = SolidityLexer(input_stream)
    lexer.removeErrorListeners()
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = SolidityParser(token_stream)
    parser.removeErrorListeners()
    tree = parser.sourceUnit()

    listener = CustomizedSolidityListener()
    walker = antlr4.ParseTreeWalker()
    walker.walk(listener, tree)
    return listener.get_contracts_info()


if __name__ == '__main__':
    # parse('./1_test.sol')
    parse_batch('./src/defihacklabs', './data/defihacklabs_loc_info.json')
