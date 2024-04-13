import importlib

def check_import(module_name:str):
    '''Check if the module is installed by trying importing, if succeed, return True, else False'''
    try:  
        importlib.import_module(module_name)
        return True
    except ImportError:  
        return False  