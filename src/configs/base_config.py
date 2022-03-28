import os, collections, copy

def path_config(username=None):
  '''
  Method to setup absolute base paths to required directories (e.g.) data,
  executalbes, outputfolder, etc

  Parameters
  ----------
  username : string, optional
    Unix username. The default is None.

  Returns
  -------
  pathconfig : DICT
    Dictionary of base-paths to the root directories (e.g. for data, executables,
                                                      output, checkpoints, etc).
  username : STRING
    Unix username which was used by the configuration script (if this
    returns 'default' - then there was no custom configuration specificed
    by the user).
  '''
  if username == None:
    username = os.environ['USER']

  paths = collections.defaultdict(dict)

  #---------------------------------------------------------------------------
  paths['default'] = {}
  paths['default']['temp_data'] = os.path.join('/scratch', os.environ['USER'], 'pipeline', 'temp_data')
  paths['default']['exec'] = os.path.join('/scratch', os.environ['USER'], 'pipeline', 'exec')
  paths['default']['output'] = os.path.join('/scratch', os.environ['USER'], 'pipeline', 'output')
  paths['default']['checkpoints'] = os.path.join('/scratch', os.environ['USER'], 'pipeline', 'checkpoints')

  #Every user can setup their own custom paths below using the default as template
  #NOTE: if you add a new key (see default above), please create a corresponding entry in the "default" dataset as
  #      well for the same key, so that it automatically is added to all configs derived from "default"
  
  #---------------------------------------------------------------------------
  # TEST'S CONFIGURATION
  #---------------------------------------------------------------------------
  test_username = 'test'
  paths[test_username] = copy.deepcopy(paths['default'])
  #make edits to this dictionary for custom paths (different from default)
  paths[test_username]['temp_data'] = os.path.join('/home', test_username, 'temp_data')
  #---------------------------------------------------------------------------

  pathconfig = {}
  if not username in paths:
    username = 'default'

  pathconfig = copy.deepcopy(paths[username])

  return pathconfig, username
