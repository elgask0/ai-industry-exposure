"""
Factor group definitions for the regression engine.

Maps factor set names (used in specification configs) to lists of column
names in the panel dataset.
"""

# Factor column mapping: set name -> list of column names in the panel
FACTOR_GROUPS = {
    'none': [],
    'mktrf': ['mktrf'],
    'mktrf_smb': ['mktrf', 'smb'],
    'mktrf_smb_hml': ['mktrf', 'smb', 'hml'],
    'mktrf_smb_hml_rmw': ['mktrf', 'smb', 'hml', 'rmw'],
    'ff3': ['mktrf', 'smb', 'hml'],
    'ff5': ['mktrf', 'smb', 'hml', 'rmw', 'cma'],
    'ff3_mom': ['mktrf', 'smb', 'hml', 'mom'],
    'ff5_mom': ['mktrf', 'smb', 'hml', 'rmw', 'cma', 'mom'],
}
