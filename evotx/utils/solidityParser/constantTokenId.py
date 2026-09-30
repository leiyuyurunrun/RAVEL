
# 48,pragma
# 170,compiler_version
# 35,import
# 121,import_statement
# 127,variable_name
# 15,contract
# 2,abstract
# 37,interface
# 40,library
# 39,is
# 94,，
# 66,using
# 28,for
# 74,{
# 75,}
# 77,;
# 30,function
# 22,event
# 13,constructor
# 43,modifier
# 50,public
# 23,external
# 38,internal
# 49,private
# 67,view
# 68,virtual
# 46,override
# 47,payable
# 51,pure
# 53,return


# overall
PRAGMA_ID = 1
COMPILER_VERSION_ID = 100
IMPORT_ID = 90
IMPORT_STATEMENT_ID = 121
VARIABLE_NAME_ID = 124
USING_ID = 66
FOR_ID = 18
LEFT_BRACKET_ID = 13
RIGHT_BRACKET_ID = 14
IS_ID = 11
COMMA_ID = 12
SEMICOLON_ID = 2

# subcontract
CONTRACT_ID = 91  # kind
INTERFACE_ID = 92
LIBRARY_ID = 93
ABSTRACT_ID = 94

# function
EVENT_ID = 27  # kind
MODIFIER_ID = 22
FUNCTION_ID = 25
CONSTRUCTOR_ID = 21
PUBLIC_ID = 116  # visibility
EXTERNAL_ID = 111
INTERNAL_ID = 113
PRIVATE_ID = 115
VIEW_ID = 119 # others
VIRTUAL_ID = 68
OVERRIDE_ID = 46
PAYABLE_ID = 114
PURE_ID = 117
RETURN_ID = 26


CLONE_TYPE=1



SUBCONTRACT_IDS = {CONTRACT_ID: 'contract', INTERFACE_ID: 'interface', LIBRARY_ID: 'library', ABSTRACT_ID: 'abstract'}
FUNCTION_IDS = {EVENT_ID: 'event', MODIFIER_ID: 'modifier', FUNCTION_ID: 'function', CONSTRUCTOR_ID: 'constructor'}
FUNCTION_VISIBILITY_IDS = {PUBLIC_ID: 'public', EXTERNAL_ID: 'external', INTERNAL_ID: 'internal', PRIVATE_ID: 'private'}