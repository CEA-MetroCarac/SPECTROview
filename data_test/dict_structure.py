df_filters = {'df0_Sheet1': {'filters': [{'expression': '1', 'state': False},
                                         {'expression': '2', 'state': True},
                                         {'expression': '3', 'state': False}]},

              'df1_Sheet1': {'filters': [{'expression': '3', 'state': False},
                                         {'expression': '4', 'state': False}]}}

df_filters = [{'expression': '1', 'state': False},
              {'expression': '2', 'state': True},
              {'expression': '3', 'state': False}
              ]

working_dfs = {'df111_Sheet1': wafer        x0_A    fwhm_E
               0   P01  403.819558  1.931026
               1   P03  403.794923  1.967943
               2   P03  403.672521  1.518390
               3   P05  403.672521  1.518390,
               'df222_Sheet1': wafer        x0_A    fwhm_E
               0   P01  403.819558  1.931026
               1   P03  403.794923  1.967943
               2   P03  403.672521  1.518390
               3   P05  403.672521  1.518390}

self.spectra = {
    wafer_2 = \
    {(0, 0):
         {'raw': {'wavenumber': [-96.698, -96.064, -95.432, -94.8],
                  'intensity': [11, 24, 33, 46]},
          'model1': {'wavenumber': [], 'intensity': []},
          'component': {'wavenumber': [], 'intensity': []},
          'residual': {'wavenumber': [], 'intensity': []}},
     (0, 1):
         {'raw': {'wavenumber': [-96.698, -96.064, -95.432, -94.8],
                  'intensity': [3, 22, 5, 26]},
          'best_fit': {'wavenumber': [], 'intensity': []},
          'component': {'wavenumber': [], 'intensity': []},
          'residual': {'wavenumber': [], 'intensity': []}},
     (2, 3): {'raw': {'wavenumber': [-96.698, -96.064, -95.432, -94.8],
                      'intensity': [11, 12, 13, 44]},
              'best_fit': {'wavenumber': [], 'intensity': []},
              'component': {'wavenumber': [], 'intensity': []},
              'residual': {'wavenumber': [], 'intensity': []}}}
    ('spectra_fs':
all
spectre         )
wafer_2 = \
    {(0, 0):
         {'raw': {'wavenumber': [-96.698, -96.064, -95.432, -94.8],
                  'intensity': [11, 24, 33, 46]},
          'model1': {'wavenumber': [], 'intensity': []},
          'component': {'wavenumber': [], 'intensity': []},
          'residual': {'wavenumber': [], 'intensity': []}},
     (0, 1):
         {'raw': {'wavenumber': [-96.698, -96.064, -95.432, -94.8],
                  'intensity': [3, 22, 5, 26]},
          'best_fit': {'wavenumber': [], 'intensity': []},
          'component': {'wavenumber': [], 'intensity': []},
          'residual': {'wavenumber': [], 'intensity': []}},
     (2, 3): {'raw': {'wavenumber': [-96.698, -96.064, -95.432, -94.8],
                      'intensity': [11, 12, 13, 44]},
              'best_fit': {'wavenumber': [], 'intensity': []},
              'component': {'wavenumber': [], 'intensity': []},
              'residual': {'wavenumber': [], 'intensity': []}}}
]
