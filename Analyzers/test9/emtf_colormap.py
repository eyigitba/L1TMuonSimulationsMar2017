def get_colormap():
  from matplotlib import cm
  from matplotlib.colors import ListedColormap
  from matplotlib.colors import LinearSegmentedColormap

  viridis_mod = ListedColormap(cm.viridis.colors, name='viridis_mod')
  viridis_mod.set_under('w',1)

  cdict = {
    'red'  : ((0.0, 0.0, 0.0), (0.746032, 0.0, 0.0), (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0, 0.0), (0.365079, 0.0, 0.0), (0.746032, 1.0, 1.0), (1.0, 1.0, 1.0)),
    'blue' : ((0.0, 0.0416, 0.0416), (0.365079, 1.0, 1.0), (1.0, 1.0, 1.0)),
  }
  blue_hot = LinearSegmentedColormap('blue_hot', cdict)

  cdict = {
    'red'  : ((0.0, 0.0, 0.0), (0.746032, 0.0, 0.0), (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0416, 0.0416), (0.365079, 1.0, 1.0), (1.0, 1.0, 1.0)),
    'blue' : ((0.0, 0.0, 0.0), (0.365079, 0.0, 0.0), (0.746032, 1.0, 1.0), (1.0, 1.0, 1.0)),
  }
  green_hot = LinearSegmentedColormap('green_hot', cdict)

  cdict = {
    'red'  : ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    'green': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
    'blue' : ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
  }
  red_binary = LinearSegmentedColormap('red_binary', cdict)

  # Update 'cm'
  cm.viridis_mod = viridis_mod
  cm.blue_hot = blue_hot
  cm.green_hot = green_hot
  cm.red_binary = red_binary
  return cm
