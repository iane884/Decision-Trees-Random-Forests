class Node:
  def __init__(self):
    self.label = None
    self.children = {}
	# you may want to add additional fields here...
    self.value = 0

  def add_child(self, value, node):
    self.children[value] = node

  def make_label(self, name):
    self.label = name